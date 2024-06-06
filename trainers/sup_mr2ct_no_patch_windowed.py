import os
from os.path import basename, dirname, join
import monai.inferers
import yaml
import time

import monai
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from shutil import copyfile
from tqdm import tqdm
import datetime
import itertools
from collections import OrderedDict

from loguru import logger
from tensorboardX import SummaryWriter

from models.generators.unet import UNet
from models.discriminators.patchgan import NLayerDiscriminator
from monai.losses.ssim_loss import SSIMLoss
from monai.metrics.regression import compute_ssim_and_cs

from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.utils import distributed_all_gather, AverageMeter, set_random_seed
from utils.visualizer import Visualizer
from utils.netdef import ShuffleUNet

from losses.gan_loss import GANLoss


class TranslationTrainer:
    def __init__(self, args, local_rank=0, tune_param=False, **kwargs):
        config_path = args.config
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        if local_rank == 0:
            if tune_param:
                config[kwargs['aug_name']] = kwargs['prob']
                if kwargs['aug_params']:
                    for k, v in kwargs['aug_params'].items():
                        config[k] = v
                
                expdir = kwargs['exp_dir_path']
                os.makedirs(expdir, exist_ok=True)
                self.expdir = expdir
                
                with open(join(self.expdir, basename(config_path)), "w") as f:
                    yaml.dump(config, f)

                ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
                ts = ts.replace(":", "_").replace("-","_")
                logger.add(os.path.join(expdir, f"{ts}.log"))
                self.visual = Visualizer(config, expdir)
                
                logger.info(f"expdir is {self.expdir}")
                
            else:
                exp_num = 1
                expdir = os.path.join(os.path.dirname(os.path.dirname(config_path)), "runs", os.path.basename(config_path).split(".")[0], str(exp_num))
                if os.path.exists(expdir):
                    while True:
                        exp_num += 1
                        expdir = os.path.join(os.path.dirname(os.path.dirname(config_path)), "runs", os.path.basename(config_path).split(".")[0], str(exp_num))
                        if not os.path.exists(expdir):
                            break
                os.makedirs(expdir, exist_ok=True)
                self.expdir = expdir
                
                copyfile(config_path, os.path.join(expdir, os.path.basename(config_path)))
                logger.add(os.path.join(expdir, "logging.log"))

                self.visual = Visualizer(config, expdir)
                
                logger.info(f"expdir is {self.expdir}")
            
        set_random_seed(config['random_seed'])
        self.config = config
        self.local_rank = local_rank
        self.ddp = args.ddp
        self.world_size = torch.cuda.device_count()
        self.resume_path = args.resume_path  # if resume_path is "", we do not need to load the checkpoint
        
        self.epoch = 0
        self.gradient_accumulation_step = args.gradient_accumulation_step
        self.amp = args.amp
        self.scaler = torch.cuda.amp.GradScaler()

        roi_size = config['patch_size']
        self.sliding_window_infer = monai.inferers.inferer.SlidingWindowInferer(
            roi_size=roi_size, sw_batch_size=1, overlap=0.5
        )
        
        # init all training ingredients needed
        self.init_model()
        self.init_criterion()
        self.init_optim()
        self.init_lr_schedule()

        if args.resume:
            if args.load_weights:
                self.load_weights(args.resume_path)
            else:
                self.load_checkpoint(args.resume_path)
        
        print("All modules have been loaded!!")
    
    def reorganize_batch(self, inputs):
        # Unbind along the first dimension
        unbound_tensors = torch.unbind(inputs, dim=0)
        # Concatenate the unbound tensors along the first dimension
        reshaped_tensor = torch.cat(unbound_tensors, dim=0)

        return reshaped_tensor
    
    def forward_G(self, source_images):
        fake_target_images = self.G(source_images)

        return fake_target_images

    def get_windowed_loss(self, fake_ct, ct_img, ct_brainmask):
        window_0_100_fake_ct = torch.clamp(fake_ct, 0, 100)
        window_0_100_ct_img = torch.clamp(ct_img, 0, 100)
        l1_loss_window_0_100 = self.criterion(window_0_100_fake_ct, window_0_100_ct_img)
        loss_ssims_window_0_100 = self.get_ssim_loss(window_0_100_fake_ct, window_0_100_ct_img, ct_brainmask)

        window_100_1500_fake_ct = torch.clamp(fake_ct, 100, 1500)
        window_100_1500_ct_img = torch.clamp(ct_img, 100, 1500)
        l1_loss_window_100_1500 = self.criterion(window_100_1500_fake_ct, window_100_1500_ct_img)
        # loss_ssims_window_100_1500 = self.get_ssim_loss(window_100_1500_fake_ct, window_100_1500_ct_img, ct_brainmask)

        loss_window_0_100 = self.config['scale_window_0_100'] * (
            self.config['lambda_l1'] * l1_loss_window_0_100 + \
            self.config['lambda_ssim_3d'] * loss_ssims_window_0_100['loss_ssim_3d'] + \
            self.config['lambda_ssim_yz'] * loss_ssims_window_0_100['loss_ssim_yz'] + \
            self.config['lambda_ssim_xz'] * loss_ssims_window_0_100['loss_ssim_xz'] + \
            self.config['lambda_ssim_xy'] * loss_ssims_window_0_100['loss_ssim_xy']
        )
    
        loss_window_100_1500 = self.config['scale_window_100_1500'] * self.config['lambda_l1'] * l1_loss_window_100_1500

        loss = loss_window_0_100 + loss_window_100_1500

        log_loss = dict()
        log_loss["loss_G/train_l1_loss_window_0_100"] = l1_loss_window_0_100.detach()
        log_loss["loss_G/train_l1_loss_window_100_1500"] = l1_loss_window_100_1500.detach()
        log_loss["loss_ssim_3d/train_window_0_100"] = loss_ssims_window_0_100['loss_ssim_3d'].detach()
        # log_loss["loss_ssim_3d/train_window_100_1500"] = loss_ssims_window_100_1500['loss_ssim_3d'].detach()
        log_loss["loss_ssim_yz/train_window_0_100"] = loss_ssims_window_0_100['loss_ssim_yz'].detach()
        # log_loss["loss_ssim_yz/train_window_100_1500"] = loss_ssims_window_100_1500['loss_ssim_yz'].detach()
        log_loss["loss_ssim_xz/train_window_0_100"] = loss_ssims_window_0_100['loss_ssim_xz'].detach()
        # log_loss["loss_ssim_xz/train_window_100_1500"] = loss_ssims_window_100_1500['loss_ssim_xz'].detach()
        log_loss["loss_ssim_xy/train_window_0_100"] = loss_ssims_window_0_100['loss_ssim_xy'].detach()
        # log_loss["loss_ssim_xy/train_window_100_1500"] = loss_ssims_window_100_1500['loss_ssim_xy'].detach()
        log_loss["loss_ssim_3d_unscaled/train_window_0_100"] = loss_ssims_window_0_100['loss_ssim_3d_unscaled'].detach()
        # log_loss["loss_ssim_3d_unscaled/train_window_100_1500"] = loss_ssims_window_100_1500['loss_ssim_3d_unscaled'].detach()
        log_loss["loss_ssim_yz_unscaled/train_window_0_100"] = loss_ssims_window_0_100['loss_ssim_yz_unscaled'].detach()
        # log_loss["loss_ssim_yz_unscaled/train_window_100_1500"] = loss_ssims_window_100_1500['loss_ssim_yz_unscaled'].detach()
        log_loss["loss_ssim_xz_unscaled/train_window_0_100"] = loss_ssims_window_0_100['loss_ssim_xz_unscaled'].detach()
        # log_loss["loss_ssim_xz_unscaled/train_window_100_1500"] = loss_ssims_window_100_1500['loss_ssim_xz_unscaled'].detach()
        log_loss["loss_ssim_xy_unscaled/train_window_0_100"] = loss_ssims_window_0_100['loss_ssim_xy_unscaled'].detach()
        # log_loss["loss_ssim_xy_unscaled/train_window_100_1500"] = loss_ssims_window_100_1500['loss_ssim_xy_unscaled'].detach()
        log_loss["loss_total/window_0_100_loss"] = loss_window_0_100.detach()
        log_loss["loss_total/window_100_1500_loss"] = loss_window_100_1500.detach()

        return {
            "loss": loss,
            "log_losses": log_loss,
            "loss_window_0_100": loss_window_0_100,
            "l1_loss_window_0_100": l1_loss_window_0_100,
            "loss_ssims_window_0_100": loss_ssims_window_0_100,
            "loss_window_100_1500": loss_window_100_1500,
            "l1_loss_window_100_1500": l1_loss_window_100_1500,
            # "loss_ssims_window_100_1500": loss_ssims_window_100_1500,
        }

    def get_ssim_loss(self, fake_ct, ct_img, ct_brainmask):
        if (fake_ct != fake_ct).sum() > 0:
            print("fake_ct is nan")
            raise Exception("fake_ct is nan")
        
        if (ct_img != ct_img).sum() > 0:
            print("ct_img is nan")
            raise Exception("ct_img is nan")
        
        if (ct_brainmask != ct_brainmask).sum() > 0:
            print("ct_brainmask is nan")
            raise Exception("ct_brainmask is nan")

        # we calculate the ssim loss only in the head mask region
        b, c, h, w, d = ct_img.shape
        # calculate 3D SSIM loss
        with torch.no_grad():
            fake_ct_min = fake_ct.view(b, -1).min(dim=1)[0]
            fake_ct_max = fake_ct.view(b, -1).max(dim=1)[0]
            fake_ct_data_range = fake_ct_max - fake_ct_min

            ct_min = ct_img.view(b, -1).min(dim=1)[0]
            ct_max = ct_img.view(b, -1).max(dim=1)[0]
            ct_data_range = ct_max - ct_min

            head_mask = ct_brainmask.type(torch.BoolTensor).cuda()
            head_mask_sum = head_mask.view(b, -1).sum(1)

            zero_mask = (fake_ct_data_range == 0) | (ct_data_range == 0)
            non_zero_mask = ~zero_mask
            n_non_zero_img = non_zero_mask.sum()
            # print("3d n_non_zero_img:", n_non_zero_img)

        norm_fake_ct = (fake_ct - fake_ct_min.view(b, 1, 1, 1, 1)) / (fake_ct_data_range.view(b, 1, 1, 1, 1) + 1)
        norm_ct_img = (ct_img - ct_min.view(b, 1, 1, 1, 1)) / (ct_data_range.view(b, 1, 1, 1, 1) + 1)

        # adaptively scale the 3d ssim loss
        ssim_3d_map, _ = compute_ssim_and_cs(
            norm_fake_ct, norm_ct_img,
            spatial_dims=3, kernel_size=(11, 11, 11), kernel_sigma=(1.5, 1.5, 1.5),
            data_range=1.0, kernel_type='gaussian', k1=0.01, k2=0.03)
        # print(ssim_3d_map.shape, norm_fake_ct.shape, norm_ct_img.shape)
        ssim_3d_map = F.pad(ssim_3d_map, (5, 5, 5, 5, 5, 5), mode='constant', value=0)
        loss_ssim_3d_map = 1 - ssim_3d_map
        loss_ssim_3d = (loss_ssim_3d_map * head_mask).reshape(b, -1).sum(1) / (head_mask_sum + 1)
        with torch.no_grad():
            loss_ssim_3d_unscaled = loss_ssim_3d_map.mean()
        loss_ssim_3d = (ct_data_range * loss_ssim_3d).mean()

        # calculate 2D SSIM loss
        def calculate_2d_ssim_loss(fake_ct_slices, ct_slices, head_mask_slices):
            # fake_ct_slices shape of (-1, h, w)
            # ct_slices shape of (-1, h, w)
            n, h, w = fake_ct_slices.shape
            with torch.no_grad():
                fake_ct_min_plane = fake_ct_slices.reshape(n, -1).min(dim=1)[0].unsqueeze(1).unsqueeze(1)
                fake_ct_max_plane = fake_ct_slices.reshape(n, -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
                fake_ct_data_range_plane = fake_ct_max_plane - fake_ct_min_plane

                ct_min_plane = ct_slices.reshape(n, -1).min(dim=1)[0].unsqueeze(1).unsqueeze(1)
                ct_max_plane = ct_slices.reshape(n, -1).max(dim=1)[0].unsqueeze(1).unsqueeze(1)
                ct_data_range_plane = ct_max_plane - ct_min_plane

                # select non-zero slices for SSIM loss to avoid zero denomiator error during normalization
                zero_mask = ct_data_range_plane == 0
                non_zero_mask = ~zero_mask.squeeze()
                n_non_zero_slices = non_zero_mask.sum()

                non_zero_head_mask_slices = head_mask_slices[non_zero_mask]
                non_zero_head_mask_slices_sum = non_zero_head_mask_slices.view(n_non_zero_slices, -1).sum(1)
                
                # if n_non_zero_slices == 0:
                #     return torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
                # print("2d n_non_zero_slices:", n_non_zero_slices)

            # print(fake_ct_slices.shape, ct_slices.shape, non_zero_mask.sum())
            norm_fake_ct_slices = (fake_ct_slices[non_zero_mask] - fake_ct_min_plane[non_zero_mask]) / (fake_ct_data_range_plane[non_zero_mask] + 1)
            norm_ct_slices = (ct_slices[non_zero_mask] - ct_min_plane[non_zero_mask]) / (ct_data_range_plane[non_zero_mask] + 1)

            # adaptively scale the 2d plane ssim loss
            norm_fake_ct_slices = norm_fake_ct_slices.unsqueeze(0)
            norm_ct_slices = norm_ct_slices.unsqueeze(0)
            ssim_plane_map, _ = compute_ssim_and_cs(
                norm_fake_ct_slices, norm_ct_slices,
                spatial_dims=2, kernel_size=(11, 11), kernel_sigma=(1.5, 1.5),
                data_range=1.0, kernel_type='gaussian', k1=0.01, k2=0.03)
            ssim_plane_map = F.pad(ssim_plane_map, (5, 5, 5, 5), mode='constant', value=0)
            
            loss_ssim_plane_map = 1 - ssim_plane_map
            loss_ssim_plane = (loss_ssim_plane_map * non_zero_head_mask_slices).view(n_non_zero_slices, -1).sum(1) / (non_zero_head_mask_slices_sum + 1)
            with torch.no_grad():
                loss_ssim_plane_unscaled = loss_ssim_plane_map.mean()
            # print(ssim_plane_map.shape, ct_data_range_plane[non_zero_mask].shape, non_zero_mask.sum(), n, loss_ssim_plane.shape)
            loss_ssim_plane = ct_data_range_plane[non_zero_mask].squeeze() * loss_ssim_plane

            return loss_ssim_plane_unscaled, loss_ssim_plane.mean()
        
        # calculate yz plane 2D SSIM loss
        fake_ct_yz_slices = fake_ct.reshape(-1, w, d)
        ct_yz_slices = ct_img.reshape(-1, w, d)
        head_mask_yz_slices = head_mask.reshape(-1, w, d)

        # adaptively scale the yz plane 2d ssim loss
        loss_ssim_yz_unscaled, loss_ssim_yz = calculate_2d_ssim_loss(fake_ct_yz_slices, ct_yz_slices, head_mask_yz_slices)
        
        # calculate xz plane 2D SSIM loss
        fake_ct_xz_slices = fake_ct.permute(0, 1, 3, 2, 4).reshape(-1, h, d)
        ct_xz_slices = ct_img.permute(0, 1, 3, 2, 4).reshape(-1, h, d)
        head_mask_xz_slices = head_mask.permute(0, 1, 3, 2, 4).reshape(-1, h, d)

        # adaptively scale the xz plane 2d ssim loss
        loss_ssim_xz_unscaled, loss_ssim_xz = calculate_2d_ssim_loss(fake_ct_xz_slices, ct_xz_slices, head_mask_xz_slices)

        # calculate xy plane 2D SSIM loss
        fake_ct_xy_slices = fake_ct.permute(0, 1, 4, 2, 3).reshape(-1, h, w)
        ct_xy_slices = ct_img.permute(0, 1, 4, 2, 3).reshape(-1, h, w)
        head_mask_xy_slices = head_mask.permute(0, 1, 4, 2, 3).reshape(-1, h, w)

        # adaptively scale the xy plane 2d ssim loss
        loss_ssim_xy_unscaled, loss_ssim_xy = calculate_2d_ssim_loss(fake_ct_xy_slices, ct_xy_slices, head_mask_xy_slices)

        return {
            "loss_ssim_3d": loss_ssim_3d,
            "loss_ssim_yz": loss_ssim_yz,
            "loss_ssim_xz": loss_ssim_xz,
            "loss_ssim_xy": loss_ssim_xy,
            "loss_ssim_3d_unscaled": loss_ssim_3d_unscaled,
            "loss_ssim_yz_unscaled": loss_ssim_yz_unscaled,
            "loss_ssim_xz_unscaled": loss_ssim_xz_unscaled,
            "loss_ssim_xy_unscaled": loss_ssim_xy_unscaled,
        }
    
    def run_train_epoch(self, train_loader):
        if self.ddp:
            train_loader.sampler.set_epoch(self.epoch)
            torch.distributed.barrier()

        self.G.train()
        
        G_loss = AverageMeter()
        ssim_loss = AverageMeter()
        window_loss = AverageMeter()

        for idx, batch in enumerate(train_loader):
            log_losses = dict()

            mr_img = batch['mr_image'].cuda(self.local_rank)
            ct_img = batch['ct_image'].cuda(self.local_rank)
            ct_brain_mask = batch['ct_brainmask']

            # ==================================
            # ======= update generators ========
            # ==================================
            self.set_requires_grad([self.G, ], requires_grad=True)

            if not self.amp:
                fake_ct = self.forward_G(mr_img,)

                loss_l1 = self.criterion(fake_ct, ct_img)

                loss_ssim_dict = self.get_ssim_loss(fake_ct, ct_img, ct_brain_mask)
                loss_ssim = self.config['lambda_ssim_3d'] * loss_ssim_dict['loss_ssim_3d'] + \
                    self.config['lambda_ssim_yz'] * loss_ssim_dict['loss_ssim_yz'] + \
                    self.config['lambda_ssim_xz'] * loss_ssim_dict['loss_ssim_xz'] + \
                    self.config['lambda_ssim_xy'] * loss_ssim_dict['loss_ssim_xy']
                
                windowed_loss_dict = self.get_windowed_loss(fake_ct, ct_img, ct_brain_mask)

                loss_G = self.config['lambda_l1'] * loss_l1 + loss_ssim + windowed_loss_dict['loss']
                
                loss_G.backward()
            else:
                with torch.cuda.amp.autocast():
                    fake_ct = self.forward_G(mr_img,)

                    loss_l1 = self.criterion_l1(fake_ct, ct_img)

                    loss_ssim_dict = self.get_ssim_loss(fake_ct, ct_img)
                    loss_ssim = self.config['lambda_ssim_3d'] * loss_ssim_dict['loss_ssim_3d'] + \
                            self.config['lambda_ssim_yz'] * loss_ssim_dict['loss_ssim_yz'] + \
                            self.config['lambda_ssim_xz'] * loss_ssim_dict['loss_ssim_xz'] + \
                            self.config['lambda_ssim_xy'] * loss_ssim_dict['loss_ssim_xy']

                    windowed_loss_dict = self.get_windowed_loss(fake_ct, ct_img, ct_brain_mask)

                    loss_G = self.config['lambda_l1'] * loss_l1 + loss_ssim + windowed_loss_dict['loss']
                    
                    self.scaler.scale(loss_G).backward()
            
            # for name, param in self.G.named_parameters():
            #     print(name)
            #     self.visual.writer.add_histogram(name, param, self.epoch * len(train_loader) + idx)
            #     self.visual.writer.add_histogram(name + "_grad", param.grad, self.epoch * len(train_loader) + idx)

            if (idx + 1) % self.gradient_accumulation_step == 0 or (idx + 1) == len(train_loader):
                if self.amp:
                    self.scaler.step(self.optimizer_G)
                    self.scaler.update()
                    self.optimizer_G.zero_grad()
                else:
                    self.optimizer_G.step()
                    self.optimizer_G.zero_grad()
            
            log_losses['loss_G/train'] = loss_l1.detach()
            log_losses['loss_ssim/train'] = loss_ssim.detach()
            log_losses['loss_ssim_3d/train'] = loss_ssim_dict['loss_ssim_3d'].detach()
            log_losses['loss_ssim_yz/train'] = loss_ssim_dict['loss_ssim_yz'].detach()
            log_losses['loss_ssim_xz/train'] = loss_ssim_dict['loss_ssim_xz'].detach()
            log_losses['loss_ssim_xy/train'] = loss_ssim_dict['loss_ssim_xy'].detach()
            log_losses['loss_ssim_3d_unscaled/train'] = loss_ssim_dict['loss_ssim_3d_unscaled'].detach()
            log_losses['loss_ssim_yz_unscaled/train'] = loss_ssim_dict['loss_ssim_yz_unscaled'].detach()
            log_losses['loss_ssim_xz_unscaled/train'] = loss_ssim_dict['loss_ssim_xz_unscaled'].detach()
            log_losses['loss_ssim_xy_unscaled/train'] = loss_ssim_dict['loss_ssim_xy_unscaled'].detach()
            log_losses['loss_total/train'] = loss_G.detach()
            log_losses.update(windowed_loss_dict['log_losses'])
                
            if (idx + 1) % self.gradient_accumulation_step == 0 or (idx + 1) == len(train_loader):
                if self.ddp:
                    loss_list = distributed_all_gather([loss_G, loss_ssim, windowed_loss_dict['loss']], out_numpy=True, is_valid=idx < len(train_loader))
                    G_loss.update(
                        np.mean(loss_list[0], axis=0), n=self.config['batch_size'] * self.world_size * self.gradient_accumulation_step
                    )
                    ssim_loss.update(
                        np.mean(loss_list[1], axis=0), n=self.config['batch_size'] * self.world_size * self.gradient_accumulation_step
                    )
                    window_loss.update(
                        np.mean(loss_list[2], axis=0), n=self.config['batch_size'] * self.world_size * self.gradient_accumulation_step
                    )

                else:
                    G_loss.update(loss_G.detach().cpu().numpy().mean(), n=self.config['batch_size'] * self.gradient_accumulation_step)
                    ssim_loss.update(loss_ssim.detach().cpu().numpy().mean(), n=self.config['batch_size'] * self.gradient_accumulation_step)
                    window_loss.update(windowed_loss_dict['loss'].detach().cpu().numpy().mean(), n=self.config['batch_size'] * self.gradient_accumulation_step)

                if self.local_rank == 0:
                    logger.info(f"Epoch {self.epoch}/{self.config['total_epochs']} {idx}/{len(train_loader)} loss_G: {G_loss.avg:.4f} loss_ssim: {ssim_loss.avg:.4f}")
                    self.visual.plot_current_errors(log_losses, idx + self.epoch * len(train_loader))
        
        self.lr_sche_G.step()
    
    @torch.no_grad()
    def run_eval_epoch(self, val_ds, mode="valid"):
        self.G.eval()

        image_num = len(val_ds)
        random_select_idx = np.random.randint(0, image_num)
        images = val_ds.__getitem__(random_select_idx)

        if monai.__version__ >= "1.0.0":
            mr_img = images['mr_image'].unsqueeze(0).cuda(self.local_rank)
            ct_img = images['ct_image'].unsqueeze(0).cuda(self.local_rank)
        else:
            mr_img = torch.from_numpy(images['mr_image']).unsqueeze(0).cuda(self.local_rank)
            ct_img = torch.from_numpy(images['ct_image']).unsqueeze(0).cuda(self.local_rank)
        
        fake_ct = self.forward_G(mr_img)

        error_image = (fake_ct - ct_img).abs()
        error_image = (error_image - error_image.min()) / (error_image.max() - error_image.min())

        return {
            "mr_img": mr_img,
            "fake_ct": fake_ct,
            "error_ct": error_image,
            "ct_img": ct_img,
        }

    @torch.no_grad()
    def run_eval_epoch_loss(self, val_ds):
        self.G.eval()

        loss = AverageMeter()
        loss_l1_ = AverageMeter()
        loss_ssim_ = AverageMeter()
        loss_ssim_3d_unscaled = AverageMeter()
        loss_ssim_yz_unscaled = AverageMeter()
        loss_ssim_xz_unscaled = AverageMeter()
        loss_ssim_xy_unscaled = AverageMeter()

        window_0_100_loss = AverageMeter()
        window_0_100_loss_l1 = AverageMeter()
        window_0_100_loss_ssim_3d_unscaled = AverageMeter()
        window_0_100_loss_ssim_yz_unscaled = AverageMeter()
        window_0_100_loss_ssim_xz_unscaled = AverageMeter()
        window_0_100_loss_ssim_xy_unscaled = AverageMeter()

        window_100_1500_loss = AverageMeter()
        window_100_1500_loss_l1 = AverageMeter()
        # window_100_1500_loss_ssim_3d_unscaled = AverageMeter()
        # window_100_1500_loss_ssim_yz_unscaled = AverageMeter()
        # window_100_1500_loss_ssim_xz_unscaled = AverageMeter()
        # window_100_1500_loss_ssim_xy_unscaled = AverageMeter()
        
        for i in tqdm(range(len(val_ds))):
            images = val_ds.__getitem__(i)

            if monai.__version__ >= "1.0.0":
                mr_img = images['mr_image'].unsqueeze(0).cuda(self.local_rank)
                ct_img = images['ct_image'].unsqueeze(0).cuda(self.local_rank)
                ct_brainmask = images['ct_brainmask'].unsqueeze(0).cuda(self.local_rank)
            else:
                mr_img = torch.from_numpy(images['mr_image']).unsqueeze(0).cuda(self.local_rank)
                ct_img = torch.from_numpy(images['ct_image']).unsqueeze(0).cuda(self.local_rank)
                ct_brainmask = torch.from_numpy(images['ct_brainmask']).unsqueeze(0).cuda(self.local_rank)
            
            fake_ct = self.forward_G(mr_img)

            error_image = (fake_ct - ct_img).abs()
            error_image = (error_image - error_image.min()) / (error_image.max() - error_image.min())

            loss_l1 = self.criterion(fake_ct, ct_img)
            loss_ssim_dict = self.get_ssim_loss(fake_ct, ct_img, ct_brainmask)
            loss_ssim = self.config['lambda_ssim_3d'] * loss_ssim_dict['loss_ssim_3d'] + \
                    self.config['lambda_ssim_yz'] * loss_ssim_dict['loss_ssim_yz'] + \
                    self.config['lambda_ssim_xz'] * loss_ssim_dict['loss_ssim_xz'] + \
                    self.config['lambda_ssim_xy'] * loss_ssim_dict['loss_ssim_xy']
            
            windowed_loss_dict = self.get_windowed_loss(fake_ct, ct_img, ct_brainmask)

            loss_G = self.config['lambda_l1'] * loss_l1 + loss_ssim + windowed_loss_dict['loss']

            loss.update(loss_G.item(), n=1)
            loss_l1_.update(loss_l1.item(), n=1)
            loss_ssim_.update(loss_ssim.item(), n=1)
            loss_ssim_3d_unscaled.update(loss_ssim_dict['loss_ssim_3d_unscaled'].item(), n=1)
            loss_ssim_yz_unscaled.update(loss_ssim_dict['loss_ssim_yz_unscaled'].item(), n=1)
            loss_ssim_xz_unscaled.update(loss_ssim_dict['loss_ssim_xz_unscaled'].item(), n=1)
            loss_ssim_xy_unscaled.update(loss_ssim_dict['loss_ssim_xy_unscaled'].item(), n=1)

            window_0_100_loss.update(windowed_loss_dict['loss_window_0_100'].item(), n=1)
            window_0_100_loss_l1.update(windowed_loss_dict['l1_loss_window_0_100'].item(), n=1)
            window_0_100_loss_ssim_3d_unscaled.update(windowed_loss_dict['loss_ssims_window_0_100']['loss_ssim_3d_unscaled'].item(), n=1)
            window_0_100_loss_ssim_yz_unscaled.update(windowed_loss_dict['loss_ssims_window_0_100']['loss_ssim_yz_unscaled'].item(), n=1)
            window_0_100_loss_ssim_xz_unscaled.update(windowed_loss_dict['loss_ssims_window_0_100']['loss_ssim_xz_unscaled'].item(), n=1)
            window_0_100_loss_ssim_xy_unscaled.update(windowed_loss_dict['loss_ssims_window_0_100']['loss_ssim_xy_unscaled'].item(), n=1)

            window_100_1500_loss.update(windowed_loss_dict['loss_window_100_1500'].item(), n=1)
            window_100_1500_loss_l1.update(windowed_loss_dict['l1_loss_window_100_1500'].item(), n=1)
            # window_100_1500_loss_ssim_3d_unscaled.update(windowed_loss_dict['loss_ssims_window_100_1500']['loss_ssim_3d_unscaled'].item(), n=1)
            # window_100_1500_loss_ssim_yz_unscaled.update(windowed_loss_dict['loss_ssims_window_100_1500']['loss_ssim_yz_unscaled'].item(), n=1)
            # window_100_1500_loss_ssim_xz_unscaled.update(windowed_loss_dict['loss_ssims_window_100_1500']['loss_ssim_xz_unscaled'].item(), n=1)
            # window_100_1500_loss_ssim_xy_unscaled.update(windowed_loss_dict['loss_ssims_window_100_1500']['loss_ssim_xy_unscaled'].item(), n=1)

        if self.local_rank == 0:
            log_losses = dict()
            log_losses['loss_G/val'] = loss_l1_.avg
            log_losses['loss_ssim/val'] = loss_ssim_.avg
            log_losses['loss_ssim_3d_unscaled/val'] = loss_ssim_3d_unscaled.avg
            log_losses['loss_ssim_yz_unscaled/val'] = loss_ssim_yz_unscaled.avg
            log_losses['loss_ssim_xz_unscaled/val'] = loss_ssim_xz_unscaled.avg
            log_losses['loss_ssim_xy_unscaled/val'] = loss_ssim_xy_unscaled.avg
            log_losses['loss_total/val'] = loss.avg

            log_losses['loss_G/val_window_0_100'] = window_0_100_loss_l1.avg
            log_losses['loss_ssim_3d_unscaled/val_window_0_100'] = window_0_100_loss_ssim_3d_unscaled.avg
            log_losses['loss_ssim_yz_unscaled/val_window_0_100'] = window_0_100_loss_ssim_yz_unscaled.avg
            log_losses['loss_ssim_xz_unscaled/val_window_0_100'] = window_0_100_loss_ssim_xz_unscaled.avg
            log_losses['loss_ssim_xy_unscaled/val_window_0_100'] = window_0_100_loss_ssim_xy_unscaled.avg
            log_losses['loss_total/val_window_0_100'] = window_0_100_loss.avg

            log_losses['loss_G/val_window_100_1500'] = window_100_1500_loss_l1.avg
            # log_losses['loss_ssim_3d_unscaled/val_window_100_1500'] = window_100_1500_loss_ssim_3d_unscaled.avg
            # log_losses['loss_ssim_yz_unscaled/val_window_100_1500'] = window_100_1500_loss_ssim_yz_unscaled.avg
            # log_losses['loss_ssim_xz_unscaled/val_window_100_1500'] = window_100_1500_loss_ssim_xz_unscaled.avg
            # log_losses['loss_ssim_xy_unscaled/val_window_100_1500'] = window_100_1500_loss_ssim_xy_unscaled.avg
            log_losses['loss_total/val_window_100_1500'] = window_100_1500_loss.avg

            self.visual.plot_current_errors(log_losses, (self.epoch + 1) * self.num_batches)

        return loss.avg

    def train(self, loaders):
        train_loader = loaders['train_loader']
        train_ds = loaders['train_ds']
        valid_ds = loaders['valid_ds']
        self.num_batches = len(train_loader)
        
        total_epoch = self.config['total_epochs']

        best_val_loss = np.inf

        for epoch in range(self.epoch + 1, total_epoch):
            self.epoch = epoch
            
            if self.local_rank == 0:
                logger.info(f"epoch {epoch} lr is {self.optimizer_G.state_dict()['param_groups'][0]['lr']}")
            
            # if self.local_rank == 0:
            #     # visualizations
            #     generated_images = self.run_eval_epoch(valid_ds)
            #     self.visualization(generated_images, (self.epoch + 1) * len(train_loader), mode="valid")

            #     val_loss = self.run_eval_epoch_loss(valid_ds)
            
            self.run_train_epoch(train_loader)
            
            if self.local_rank == 0 and (epoch + 1) % self.config["save_checkpoint_freq"] == 0:
                # visualizations
                generated_images = self.run_eval_epoch(train_ds)
                self.visualization(generated_images, (self.epoch + 1) * len(train_loader), mode="train")
                generated_images = self.run_eval_epoch(valid_ds)
                self.visualization(generated_images, (self.epoch + 1) * len(train_loader), mode="valid")

                val_loss = self.run_eval_epoch_loss(valid_ds)
                logger.info(f"Epoch {self.epoch}/{self.config['total_epochs']} {len(train_loader)}/{len(train_loader)} val loss: {val_loss:.4f}, Best val loss: {best_val_loss:.4f}")
                # self.visual.writer.add_scalar("val_loss", val_loss, self.epoch * len(train_loader))

                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    logger.info(f"Best val loss: {best_val_loss:.4f}")
                    self.save_checkpoint("model_best.pt")

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def init_model(self,):
        spatial_dim = self.config['spatial_dim']
        in_channels = self.config['in_channel']
        model_name = self.config['model_name']

        if model_name == "shuffleunet":
            img_size = self.config['img_centercrop_size']
            transformer_layers = self.config['transformer_layers']
            num_residual_units = self.config['num_residual_units']
            shuffleunet_filters = self.config['shuffleunet_filters']
            self.G = ShuffleUNet(dimensions=3, in_channels=1, out_channels=1,
                    channels=shuffleunet_filters,
                    strides=(2, 2, 2, 2),
                    kernel_size=3, up_kernel_size=3, num_res_units=num_residual_units, img_size=img_size, 
                    transformer_layers=transformer_layers)
        elif model_name == "unet":
            strides = self.config['strides']
            filters = self.config['filters'][: len(strides)]
            kernel_size = self.config['kernel_size']
            unet_use_resblock = self.config['unet_use_resblock']
            upsample_kernel_size = strides[1:]
            self.G = UNet(in_channels, in_channels, filters, strides, kernel_size, upsample_kernel_size, use_resblock=unet_use_resblock)

        # Count parameters in each layer
        total_params = sum(p.numel() for p in self.G.parameters() if p.requires_grad)

        # Report the total number of trainable parameters
        print("Total Trainable Parameters:", total_params)

        if self.ddp and self.world_size > 1:
            self.G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.G_S)
            self.G = self.G.cuda(self.local_rank)

            self.G = torch.nn.parallel.DistributedDataParallel(
                self.G, device_ids=[self.local_rank], find_unused_parameters=False
            )
        else:
            self.G = self.G.cuda(self.local_rank)
    
    def init_optim(self,):
        g_name = self.config['generator']['name']
        g_lr = self.config['generator']['base_lr']
        g_wd = self.config['generator']['weight_decay']
        g_betas = self.config['generator']['betas']
        if g_name == "sgd":
            self.optimizer_G = torch.optim.SGD(
                self.G.parameters(), 
                lr=g_lr, momentum=0.9, weight_decay=g_wd
            )
        elif g_name == "adamw":
            self.optimizer_G = torch.optim.AdamW(
                self.G.parameters(), 
                lr=g_lr, betas=g_betas, weight_decay=g_wd
            )
        elif g_name == "rangerlars":
            from utils.optimizer import RangerLars
            self.optimizer_G = RangerLars(
                self.G.parameters(), betas=g_betas,
                lr=g_lr, weight_decay=g_wd
            )
        else:
            raise Exception("No optimizer is selected!!")
    
    def init_lr_schedule(self,):
        g_warmup_epochs = self.config['generator']['warmup_epochs']
        total_epochs = self.config['total_epochs']
        self.lr_sche_G = LinearWarmupCosineAnnealingLR(self.optimizer_G, warmup_epochs=g_warmup_epochs, max_epochs=total_epochs)
        # first step the lr_scheduler to make the lr to the initial value, not 0 lr
        self.lr_sche_G.step()
    
    def init_criterion(self, ):
        self.criterion_mse = nn.MSELoss()
        self.criterion_l1 = nn.L1Loss()

        self.criterion = nn.L1Loss() if self.config['criterion'] == "l1" else nn.MSELoss()
        self.criterion_ssim_3d = SSIMLoss(spatial_dims=3, data_range=1)  # for whole image ssim loss
        self.criterion_ssim_2d = SSIMLoss(spatial_dims=2, data_range=1, reduction="none")  # calculating ssim loss for each slice as then averaging them
    
    @torch.no_grad()
    def visualization(self, generated_images_dict, step, mode):
        visual_s_list = []
        visual_t_list = []

        mr_images = (generated_images_dict['mr_img'] + 1) / 2
        ct_images = (generated_images_dict['ct_img'] + 1024) / 3000
        fake_ct = (generated_images_dict['fake_ct'] + 1024) / 3000
        error_image = generated_images_dict['error_ct']

        mr_center_slice_num = mr_images.shape[-1] // 2
        ct_center_slice_num = ct_images.shape[-1] // 2


        visual_s_list += [
            ('real_mr', mr_images[..., mr_center_slice_num].cpu()), 
            ('fake_ct', fake_ct[..., ct_center_slice_num].cpu()), 
            ('error_ct', error_image[..., ct_center_slice_num].cpu()),
            ('real_ct', ct_images[..., ct_center_slice_num].cpu()), 
        ]
        self.visual.display_current_results(OrderedDict(visual_s_list), "s", step, mode)

    def load_weights(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        logger.info(f"Load checkpoint from {checkpoint_path}")
        G_state_dict = state_dict['G']

        if not self.ddp:
            self.G.load_weights(G_state_dict)
        else:
            self.G.module.load_weights(G_state_dict)

        # self.G.load_state_dict(G_state_dict)

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        logger.info(f"Load checkpoint from {checkpoint_path}")
        G_state_dict = state_dict['G']

        self.G.load_state_dict(G_state_dict)

        self.optimizer_G.load_state_dict(state_dict['optimizer_G'])

        if self.lr_sche_G is not None:
            self.lr_sche_G.load_state_dict(state_dict['scheduler_G'])
        
        self.epoch = state_dict['epoch']

    def save_checkpoint(self, model_name):
        G_state_dict = self.G.state_dict() if not self.ddp else self.G.module.state_dict()
        save_dict = {
            "epoch": self.epoch, 
            "G": G_state_dict,
        }
        if self.optimizer_G is not None:
            save_dict["optimizer_G"] = self.optimizer_G.state_dict()
        if self.lr_sche_G is not None:
            save_dict["scheduler_G"] = self.lr_sche_G.state_dict()

        model_name = os.path.join(self.expdir, model_name)
        torch.save(save_dict, model_name)