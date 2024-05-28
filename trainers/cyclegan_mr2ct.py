import os
from os.path import basename, dirname, join
import yaml
import time

import monai
import torch
import torch.nn as nn
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

from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.utils import distributed_all_gather, AverageMeter, set_random_seed
from utils.visualizer import Visualizer

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
                if args.resume:
                    expdir = args.resume_path
                    self.expdir = expdir
                    logger.add(os.path.join(expdir, "logging.log"))
                    self.writer = SummaryWriter(expdir)
                    
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
        
        # init all training ingredients needed
        self.init_model()
        self.init_criterion()
        self.init_optim()
        self.init_lr_schedule()

        # cyclegan img cache pool
        self.num_fakes = 0
        self.fake_T_images_pool = torch.zeros(
            (
                config["cyclegan_image_pool_size"] * args.gradient_accumulation_step, 
                config["batch_size"], 1,   # 1 is the channel number
                *config["ct_final_img_size"]
            )
        )
        self.fake_S_images_pool = torch.zeros(
            (
                config["cyclegan_image_pool_size"] * args.gradient_accumulation_step, 
                config["batch_size"], 1,    # 1 is the channel number
                *config["mr_final_img_size"]
            )
        )
        # print(self.fake_T_images_pool.shape, self.fake_S_images_pool.shape)

        if args.resume:
            self.load_checkpoint(join(args.resume_path, "model_last.pt"))
        
        print("All modules have been loaded!!")
    
    def interpolate_like_size(self, tensor, size):
        return nn.functional.interpolate(tensor, size=size, mode="trilinear", align_corners=False)
    
    def forward_G_S(self, source_images, target_size):
        fake_target_images = self.G_S(source_images)
        fake_target_images = self.interpolate_like_size(fake_target_images, target_size)
        fake_target_images = torch.tanh(fake_target_images)

        return fake_target_images

    def forward_G_T(self, target_images, source_size):
        fake_source_images = self.G_T(target_images)
        fake_source_images = self.interpolate_like_size(fake_source_images, source_size)
        fake_source_images = torch.tanh(fake_source_images)

        return fake_source_images

    def run_train_epoch(self, train_loader):
        if self.ddp:
            train_loader.sampler.set_epoch(self.epoch)
            torch.distributed.barrier()

        self.G_S.train()
        self.G_T.train()
        self.discriminator_S.train()
        self.discriminator_T.train()
        
        G_loss = AverageMeter()
        ADV_loss = AverageMeter()
        D_loss = AverageMeter()

        for idx, batch in enumerate(train_loader):
            log_losses = dict()

            mr_img = batch['mr_image'].cuda(self.local_rank)
            ct_img = batch['ct_image'].cuda(self.local_rank)

            # ==================================
            # ======= update generators ========
            # ==================================
            self.set_requires_grad([self.G_S, self.G_T], requires_grad=True)
            self.set_requires_grad([self.discriminator_S, self.discriminator_T], requires_grad=False)

            if not self.amp:
                fake_ct = self.forward_G_S(mr_img, ct_img.shape[2:])
                fake_mr = self.forward_G_T(ct_img, mr_img.shape[2:])

                cyc_mr = self.forward_G_T(fake_ct, mr_img.shape[2:])
                cyc_ct = self.forward_G_S(fake_mr, ct_img.shape[2:])

                if self.config['lambda_id'] > 0:
                    id_mr = self.forward_G_T(mr_img, mr_img.shape[2:])
                    id_ct = self.forward_G_S(ct_img, ct_img.shape[2:])
                
                loss_gan_mr = self.criterion_gan(self.discriminator_S(fake_mr), True)
                loss_gan_ct = self.criterion_gan(self.discriminator_T(fake_ct), True)
                loss_gan = (loss_gan_mr + loss_gan_ct) * 0.5
                loss_G = loss_gan * self.config['lambda_gan']

                loss_cyc_mr = self.criterion_cycle(cyc_mr, mr_img)
                loss_cyc_ct = self.criterion_cycle(cyc_ct, ct_img)
                loss_cyc = (loss_cyc_mr + loss_cyc_ct) * 0.5
                loss_G += loss_cyc * self.config['lambda_cyc']

                if self.config['lambda_id'] > 0:
                    loss_id_mr = self.criterion_identity(id_mr, mr_img)
                    loss_id_ct = self.criterion_identity(id_ct, ct_img)
                    loss_identity = (loss_id_mr + loss_id_ct) * 0.5
                    loss_G += loss_identity * self.config['lambda_id']
                
                loss_G.backward()
            else:
                with torch.cuda.amp.autocast():
                    fake_ct = self.forward_G_S(mr_img, ct_img.shape[2:])
                    fake_mr = self.forward_G_T(ct_img, mr_img.shape[2:])

                    cyc_mr = self.forward_G_T(fake_ct, mr_img.shape[2:])
                    cyc_ct = self.forward_G_S(fake_mr, ct_img.shape[2:])

                    if self.config['lambda_id'] > 0:
                        id_mr = self.forward_G_T(mr_img, mr_img.shape[2:])
                        id_ct = self.forward_G_S(ct_img, ct_img.shape[2:])

                    loss_gan_mr = self.criterion_gan(self.discriminator_S(fake_mr), True)
                    loss_gan_ct = self.criterion_gan(self.discriminator_T(fake_ct), True)
                    loss_gan = (loss_gan_mr + loss_gan_ct) * 0.5
                    loss_G = loss_gan * self.config['lambda_gan']

                    loss_cyc_mr = self.criterion_cycle(cyc_mr, mr_img)
                    loss_cyc_ct = self.criterion_cycle(cyc_ct, ct_img)
                    loss_cyc = (loss_cyc_mr + loss_cyc_ct) * 0.5
                    loss_G += loss_cyc * self.config['lambda_cyc']

                    if self.config['lambda_id'] > 0:
                        loss_id_mr = self.criterion_identity(id_mr, mr_img)
                        loss_id_ct = self.criterion_identity(id_ct, ct_img)
                        loss_identity = (loss_id_mr + loss_id_ct) * 0.5
                        loss_G += loss_identity * self.config['lambda_id']

                    self.scaler.scale(loss_G).backward()
            
            fake_ct_from_pool = self.fake_images_pool(self.num_fakes, fake_ct.detach().cpu(), self.fake_T_images_pool).cuda(self.local_rank)
            fake_mr_from_pool = self.fake_images_pool(self.num_fakes, fake_mr.detach().cpu(), self.fake_S_images_pool).cuda(self.local_rank)
            self.num_fakes += 1

            if (idx + 1) % self.gradient_accumulation_step == 0 or (idx + 1) == len(train_loader):
                if self.amp:
                    self.scaler.step(self.optimizer_G)
                    self.scaler.update()
                    self.optimizer_G.zero_grad()
                else:
                    self.optimizer_G.step()
                    self.optimizer_G.zero_grad()
            
            log_losses["generator/loss_gan_mr"] = loss_gan_mr.detach()
            log_losses["generator/loss_gan_ct"] = loss_gan_ct.detach()
            log_losses["generator/loss_gan"] = loss_gan.detach()
            log_losses["generator/loss_cyc_mr"] = loss_cyc_mr.detach()
            log_losses["generator/loss_cyc_ct"] = loss_cyc_ct.detach()
            log_losses["generator/loss_cyc"] = loss_cyc.detach()
            if self.config['lambda_id'] > 0:
                log_losses["generator/identity_loss_mr"] = loss_id_mr.detach()
                log_losses["generator/identity_loss_ct"] = loss_id_ct.detach()
                log_losses["generator/identity_loss"] = loss_identity.detach()

            # ======================================
            # ======= update discriminators ========
            # ======================================
            self.set_requires_grad([self.G_S, self.G_T], requires_grad=False)
            self.set_requires_grad([self.discriminator_S, self.discriminator_T], requires_grad=True)

            if not self.amp:
                loss_d_ct = self.criterion_gan(self.discriminator_T(ct_img), True) + \
                    self.criterion_gan(self.discriminator_T(fake_ct_from_pool.detach()), False)
                loss_d_mr = self.criterion_gan(self.discriminator_S(mr_img), True) + \
                    self.criterion_gan(self.discriminator_S(fake_mr_from_pool.detach()), False)
                loss_D = (loss_d_ct + loss_id_mr) * 0.5

                loss_D.backward()
            else:
                with torch.cuda.amp.autocast():
                    loss_d_ct = self.criterion_gan(self.discriminator_T(ct_img), True) + \
                        self.criterion_gan(self.discriminator_T(fake_ct_from_pool), False)
                    loss_d_mr = self.criterion_gan(self.discriminator_S(mr_img), True) + \
                        self.criterion_gan(self.discriminator_S(fake_mr_from_pool), False)
                    loss_D = (loss_d_ct + loss_d_mr) * 0.5

                    self.scaler.scale(loss_D).backward()


            if (idx + 1) % self.gradient_accumulation_step == 0 or (idx + 1) == len(train_loader):
                if self.amp:
                    self.scaler.step(self.optimizer_D)
                    self.scaler.update()
                    self.optimizer_D.zero_grad()
                else:
                    self.optimizer_D.step()
                    self.optimizer_D.zero_grad()

                log_losses["discriminator/loss_d_ct"] = loss_d_ct.detach()
                log_losses["discriminator/loss_d_mr"] = loss_d_mr.detach()
                log_losses["discriminator/loss_D"] = loss_D.detach()
                
            
            if (idx + 1) % self.gradient_accumulation_step == 0 or (idx + 1) == len(train_loader):
                if self.ddp:
                    loss_list = distributed_all_gather([loss_G, loss_gan, loss_D], out_numpy=True, is_valid=idx < len(train_loader))
                    G_loss.update(
                        np.mean(loss_list[0], axis=0), n=self.config['batch_size'] * self.world_size * self.gradient_accumulation_step
                    )
                    ADV_loss.update(
                        np.mean(loss_list[1], axis=0), n=self.config['batch_size'] * self.world_size * self.gradient_accumulation_step
                    )
                    D_loss.update(
                        np.mean(loss_list[2], axis=0), n=self.config['batch_size'] * self.world_size * self.gradient_accumulation_step
                    )

                else:
                    G_loss.update(loss_G.detach().cpu().numpy().mean(), n=self.config['batch_size'] * self.gradient_accumulation_step)
                    ADV_loss.update(loss_gan.detach().cpu().numpy().mean(), n=self.config['batch_size'] * self.gradient_accumulation_step)
                    D_loss.update(loss_D.detach().cpu().numpy().mean(), n=self.config['batch_size'] * self.gradient_accumulation_step)
                
                if self.local_rank == 0:
                    logger.info(f"Epoch {self.epoch}/{self.config['total_epochs']} {idx}/{len(train_loader)} loss_G: {G_loss.avg:.4f} ADV_loss: {ADV_loss.avg:.4f} loss_D: {D_loss.avg:.4f}")
                    self.visual.plot_current_errors(log_losses, idx + self.epoch * len(train_loader))
        
        self.lr_sche_G.step()
        self.lr_sche_D.step()
    
    @torch.no_grad()
    def run_eval_epoch(self, val_ds):
        self.G_S.eval()
        self.G_T.eval()
        self.discriminator_S.eval()
        self.discriminator_T.eval()

        image_num = len(val_ds)
        random_select_idx = np.random.randint(0, image_num)
        images = val_ds.__getitem__(random_select_idx)

        if monai.__version__ >= "1.0.0":
            mr_img = images['mr_image'].unsqueeze(0).cuda(self.local_rank)
            ct_img = images['ct_image'].unsqueeze(0).cuda(self.local_rank)
        else:
            mr_img = torch.from_numpy(images['mr_image']).unsqueeze(0).cuda(self.local_rank)
            ct_img = torch.from_numpy(images['ct_image']).unsqueeze(0).cuda(self.local_rank)

        fake_ct = self.G_S(mr_img)
        fake_ct = self.interpolate_like_size(fake_ct, ct_img.shape[2:])
        fake_mr = self.G_T(ct_img)
        fake_mr = self.interpolate_like_size(fake_mr, mr_img.shape[2:])

        cyc_mr = self.G_T(fake_ct)
        cyc_mr = self.interpolate_like_size(cyc_mr, mr_img.shape[2:])
        cyc_ct = self.G_S(fake_mr)
        cyc_ct = self.interpolate_like_size(cyc_ct, ct_img.shape[2:])

        id_mr = None
        id_ct = None
        if self.config['lambda_id'] > 0:
            id_mr = self.G_T(mr_img)
            id_mr = self.interpolate_like_size(id_mr, mr_img.shape[2:])
            id_ct = self.G_S(ct_img)
            id_ct = self.interpolate_like_size(id_ct, ct_img.shape[2:])

        return {
            "mr_img": mr_img,
            "ct_img": ct_img,
            "fake_ct": fake_ct,
            "fake_mr": fake_mr,
            "cyc_mr": cyc_mr,
            "cyc_ct": cyc_ct,
            "id_mr": id_mr,
            "id_ct": id_ct
        }
    
    def train(self, loaders):
        train_loader = loaders['train_loader']
        train_ds = loaders['train_ds']
        valid_ds = loaders['valid_ds']
        
        total_epoch = self.config['total_epochs']

        for epoch in range(self.epoch + 1, total_epoch):
            self.epoch = epoch
            
            if self.local_rank == 0:
                logger.info(f"epoch {epoch} G lr is {self.optimizer_G.state_dict()['param_groups'][0]['lr']}, D lr is {self.optimizer_D.state_dict()['param_groups'][0]['lr']}")
            
            self.run_train_epoch(train_loader)
            
            if self.local_rank == 0 and (epoch + 1) % self.config["save_checkpoint_freq"] == 0:
                logger.info("Saving checkpoint...")
                self.save_checkpoint(f"model_{self.epoch}.pt")

                # visualizations
                generated_images = self.run_eval_epoch(train_ds)
                self.visualization(generated_images, (self.epoch + 1) * len(train_loader), mode="train")
                generated_images = self.run_eval_epoch(valid_ds)
                self.visualization(generated_images, (self.epoch + 1) * len(train_loader), mode="valid")

    def fake_images_pool(self, num_fakes, fake, fake_pool):
        fake_pool_size = fake_pool.shape[0]
        if num_fakes < fake_pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, fake_pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

    def cache_images_for_accumulation(self, cache, images, idx):
        cache[idx] = images.detach().cpu()

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

        strides = self.config['strides']
        filters = self.config['filters'][: len(strides)]
        kernel_size = self.config['kernel_size']
        upsample_kernel_size = strides[1:]
        self.G_S = UNet(in_channels, in_channels, filters, strides, kernel_size, upsample_kernel_size)
        self.G_T = UNet(in_channels, in_channels, filters, strides, kernel_size, upsample_kernel_size)

        n_layers = self.config['n_layers']
        ndf = self.config['ndf']
        norm_type = nn.BatchNorm3d if self.config['norm_type'] == "batch" else nn.InstanceNorm3d
        self.discriminator_S = NLayerDiscriminator(in_channels, ndf=ndf, n_layers=n_layers, norm_layer=norm_type)
        self.discriminator_T = NLayerDiscriminator(in_channels, ndf=ndf, n_layers=n_layers, norm_layer=norm_type)
                
        # pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print("Total parameters count", pytorch_total_params)
        
        if self.ddp and self.world_size > 1:
            self.G_S = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.G_S)
            self.G_S = self.G_S.cuda(self.local_rank)

            self.G_T = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.G_T)
            self.G_T = self.G_T.cuda(self.local_rank)

            self.discriminator_S = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator_S)
            self.discriminator_S = self.discriminator_S.cuda(self.local_rank)

            self.discriminator_T = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator_T)
            self.discriminator_T = self.discriminator_T.cuda(self.local_rank)

            self.G_S = torch.nn.parallel.DistributedDataParallel(
                self.G_S, device_ids=[self.local_rank], find_unused_parameters=False
            )
            self.G_T = torch.nn.parallel.DistributedDataParallel(
                self.G_T, device_ids=[self.local_rank], find_unused_parameters=False
            )
            self.discriminator_S = torch.nn.parallel.DistributedDataParallel(
                self.discriminator_S, device_ids=[self.local_rank], find_unused_parameters=False
            )
            self.discriminator_T = torch.nn.parallel.DistributedDataParallel(
                self.discriminator_T, device_ids=[self.local_rank], find_unused_parameters=False
            )
        else:
            self.G_S = self.G_S.cuda(self.local_rank)
            self.G_T = self.G_T.cuda(self.local_rank)
            self.discriminator_S = self.discriminator_S.cuda(self.local_rank)
            self.discriminator_T = self.discriminator_T.cuda(self.local_rank)
    
    def init_optim(self,):
        g_lr = self.config['generator']['base_lr']
        g_wd = self.config['generator']['weight_decay']
        g_betas = self.config['generator']['betas']
        self.optimizer_G = torch.optim.AdamW(
            itertools.chain(self.G_S.parameters(), self.G_T.parameters()), 
            lr=g_lr, betas=g_betas, weight_decay=g_wd
        )
        d_lr = self.config['discriminator']['base_lr']
        d_wd = self.config['discriminator']['weight_decay']
        d_betas = self.config['discriminator']['betas']
        self.optimizer_D = torch.optim.AdamW(
            itertools.chain(self.discriminator_S.parameters(), self.discriminator_T.parameters()), 
            lr=d_lr, betas=d_betas, weight_decay=d_wd
        )
    
    def init_lr_schedule(self,):
        g_warmup_epochs = self.config['generator']['warmup_epochs']
        d_warmup_epochs = self.config['discriminator']['warmup_epochs']
        total_epochs = self.config['total_epochs']
        self.lr_sche_G = LinearWarmupCosineAnnealingLR(self.optimizer_G, warmup_epochs=g_warmup_epochs, max_epochs=total_epochs)
        self.lr_sche_D = LinearWarmupCosineAnnealingLR(self.optimizer_D, warmup_epochs=d_warmup_epochs, max_epochs=total_epochs)
        # first step the lr_scheduler to make the lr to the initial value, not 0 lr
        self.lr_sche_G.step()
        self.lr_sche_D.step()
    
    def init_criterion(self, ):
        self.criterion_gan = GANLoss()  # default is LSGAN
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
    
    @torch.no_grad()
    def visualization(self, generated_images_dict, step, mode):
        visual_s_list = []
        visual_t_list = []

        mr_images = generated_images_dict['mr_img']
        ct_images = generated_images_dict['ct_img']
        fake_mr = generated_images_dict['fake_mr']
        fake_ct = generated_images_dict['fake_ct']
        cyc_mr = generated_images_dict['cyc_mr']
        cyc_ct = generated_images_dict['cyc_ct']

        mr_center_slice_num = mr_images.shape[-1] // 2
        ct_center_slice_num = ct_images.shape[-1] // 2


        visual_s_list += [
            ('real_mr', mr_images[..., mr_center_slice_num].cpu()), 
            ('fake_ct', fake_ct[..., ct_center_slice_num].cpu()), 
            ('cyc_mr', cyc_mr[..., mr_center_slice_num].cpu())
        ]
        visual_t_list += [
            ('real_ct', ct_images[..., ct_center_slice_num].cpu()), 
            ('fake_mr', fake_mr[..., mr_center_slice_num].cpu()), 
            ('cyc_ct', cyc_ct[..., ct_center_slice_num].cpu())
        ]

        if self.config['lambda_id'] > 0:
            id_mr = generated_images_dict['id_mr']
            id_ct = generated_images_dict['id_ct']
            visual_s_list += [
                ('id_mr', id_mr[..., mr_center_slice_num].cpu())
            ]
            visual_t_list += [
                ('id_ct', id_ct[..., ct_center_slice_num].cpu())
            ]
        
        self.visual.display_current_results(OrderedDict(visual_s_list), "s", step, mode)
        self.visual.display_current_results(OrderedDict(visual_t_list), "t", step, mode)

    def load_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        logger.info(f"Load checkpoint from {checkpoint_path}")
        G_S_state_dict = state_dict['G_S']
        G_T_state_dict = state_dict['G_T']
        discriminator_S_state_dict = state_dict['discriminator_S']
        discriminator_T_state_dict = state_dict['discriminator_T']

        self.G_S.load_state_dict(G_S_state_dict)
        self.G_T.load_state_dict(G_T_state_dict)
        self.discriminator_S.load_state_dict(discriminator_S_state_dict)
        self.discriminator_T.load_state_dict(discriminator_T_state_dict)

        self.optimizer_G.load_state_dict(state_dict['optimizer_G'])
        self.optimizer_D.load_state_dict(state_dict['optimizer_D'])

        if self.lr_sche_G is not None:
            self.lr_sche_G.load_state_dict(state_dict['scheduler_G'])
        if self.lr_sche_D is not None:
            self.lr_sche_D.load_state_dict(state_dict['scheduler_D'])
        
        self.epoch = state_dict['epoch']

    def save_checkpoint(self, model_name):
        G_S_state_dict = self.G_S.state_dict() if not self.ddp else self.G_S.module.state_dict()
        G_T_state_dict = self.G_T.state_dict() if not self.ddp else self.G_T.module.state_dict()
        discriminator_S_state_dict = self.discriminator_S.state_dict() if not self.ddp else self.discriminator_S.module.state_dict()
        discriminator_T_state_dict = self.discriminator_T.state_dict() if not self.ddp else self.discriminator_T.module.state_dict()
        save_dict = {
            "epoch": self.epoch, 
            "G_S": G_S_state_dict,
            "G_T": G_T_state_dict,
            "discriminator_S": discriminator_S_state_dict,
            "discriminator_T": discriminator_T_state_dict,
        }
        if self.optimizer_G is not None:
            save_dict["optimizer_G"] = self.optimizer_G.state_dict()
        if self.optimizer_D is not None:
            save_dict["optimizer_D"] = self.optimizer_D.state_dict()
        if self.lr_sche_G is not None:
            save_dict["scheduler_G"] = self.lr_sche_G.state_dict()
        if self.lr_sche_D is not None:
            save_dict["scheduler_D"] = self.lr_sche_D.state_dict()

        model_name = os.path.join(self.expdir, model_name)
        torch.save(save_dict, model_name)