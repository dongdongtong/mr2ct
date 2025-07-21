import os
from os.path import basename, dirname, join
import monai.inferers
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
                if args.resume and not args.load_weights:
                    expdir = dirname(args.resume_path)
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

    def run_train_epoch(self, train_loader):
        if self.ddp:
            train_loader.sampler.set_epoch(self.epoch)
            torch.distributed.barrier()

        self.G.train()
        
        G_loss = AverageMeter()

        for idx, batch in enumerate(train_loader):
            log_losses = dict()

            mr_img = batch['mr_image'].cuda(self.local_rank)
            ct_img = batch['ct_image'].cuda(self.local_rank)

            # ==================================
            # ======= update generators ========
            # ==================================
            self.set_requires_grad([self.G, ], requires_grad=True)

            if not self.amp:
                fake_ct = self.forward_G(mr_img,)

                loss_G = self.criterion(fake_ct, ct_img)
                
                loss_G.backward()
            else:
                with torch.cuda.amp.autocast():
                    fake_ct = self.forward_G(mr_img,)

                    loss_G = self.criterion(fake_ct, ct_img)

                    self.scaler.scale(loss_G).backward()

            if (idx + 1) % self.gradient_accumulation_step == 0 or (idx + 1) == len(train_loader):
                if self.amp:
                    self.scaler.step(self.optimizer_G)
                    self.scaler.update()
                    self.optimizer_G.zero_grad()
                else:
                    self.optimizer_G.step()
                    self.optimizer_G.zero_grad()
            
            log_losses["loss_G"] = loss_G.detach()
                
            if (idx + 1) % self.gradient_accumulation_step == 0 or (idx + 1) == len(train_loader):
                if self.ddp:
                    loss_list = distributed_all_gather([loss_G, ], out_numpy=True, is_valid=idx < len(train_loader))
                    G_loss.update(
                        np.mean(loss_list[0], axis=0), n=self.config['batch_size'] * self.world_size * self.gradient_accumulation_step
                    )

                else:
                    G_loss.update(loss_G.detach().cpu().numpy().mean(), n=self.config['batch_size'] * self.gradient_accumulation_step)

                if self.local_rank == 0:
                    logger.info(f"Epoch {self.epoch}/{self.config['total_epochs']} {idx}/{len(train_loader)} loss_G: {G_loss.avg:.4f}")
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
        
        if mode == "train":
            # print(mr_img.shape, ct_img.shape)
            mr_img = mr_img[0, 0:1]
            ct_img = ct_img[0, 0:1]
        
        # print(mr_img.shape)
        fake_ct = self.sliding_window_infer(mr_img, self.G)

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
        for i in tqdm(range(len(val_ds))):
            images = val_ds.__getitem__(i)

            if monai.__version__ >= "1.0.0":
                mr_img = images['mr_image'].unsqueeze(0).cuda(self.local_rank)
                ct_img = images['ct_image'].unsqueeze(0).cuda(self.local_rank)
            else:
                mr_img = torch.from_numpy(images['mr_image']).unsqueeze(0).cuda(self.local_rank)
                ct_img = torch.from_numpy(images['ct_image']).unsqueeze(0).cuda(self.local_rank)
            
            fake_ct = self.sliding_window_infer(mr_img, self.G)

            error_image = (fake_ct - ct_img).abs()
            error_image = (error_image - error_image.min()) / (error_image.max() - error_image.min())

            loss.update(self.criterion(fake_ct, ct_img).item(), n=1)

        return loss.avg

    def train(self, loaders):
        train_loader = loaders['train_loader']
        train_ds = loaders['train_ds']
        valid_ds = loaders['valid_ds']
        
        total_epoch = self.config['total_epochs']

        best_val_loss = np.inf

        for epoch in range(self.epoch + 1, total_epoch):
            self.epoch = epoch
            
            if self.local_rank == 0:
                logger.info(f"epoch {epoch} lr is {self.optimizer_G.state_dict()['param_groups'][0]['lr']}")
            
            self.run_train_epoch(train_loader)
            
            if self.local_rank == 0 and (epoch + 1) % self.config["save_checkpoint_freq"] == 0:
                # visualizations
                generated_images = self.run_eval_epoch(train_ds, mode="train")
                self.visualization(generated_images, (self.epoch + 1) * len(train_loader), mode="train")
                generated_images = self.run_eval_epoch(valid_ds)
                self.visualization(generated_images, (self.epoch + 1) * len(train_loader), mode="valid")

                val_loss = self.run_eval_epoch_loss(valid_ds)
                logger.info(f"Epoch {self.epoch}/{self.config['total_epochs']} {len(train_loader)}/{len(train_loader)} val loss: {val_loss:.4f}, Best val loss: {best_val_loss:.4f}")
                self.visual.writer.add_scalar("val_loss", val_loss, self.epoch * len(train_loader))

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
            shuffleunet_strides = self.config.get('shuffleunet_strides', (2, 2, 2, 2))
            self.G = ShuffleUNet(dimensions=3, in_channels=1, out_channels=1,
                    channels=shuffleunet_filters, 
                    strides=shuffleunet_strides,
                    kernel_size = 3, up_kernel_size = 3, num_res_units=num_residual_units, 
                    img_size=img_size, transformer_layers=transformer_layers)
            print(self.G)
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
    
    @torch.no_grad()
    def visualization(self, generated_images_dict, step, mode):
        visual_s_list = []
        visual_t_list = []

        mr_images = (generated_images_dict['mr_img'] + 1) / 2
        window_0_100_fake_ct = torch.clamp(generated_images_dict['fake_ct'], 0, 100) / 100
        window_0_100_ct_images = torch.clamp(generated_images_dict['ct_img'], 0, 100) / 100
        window_100_1500_fake_ct = (torch.clamp(generated_images_dict['fake_ct'], 100, 1500)) / 1500
        window_100_1500_ct_images = (torch.clamp(generated_images_dict['ct_img'], 100, 1500)) / 1500
        ct_images = (generated_images_dict['ct_img'] + 1024) / 4024
        fake_ct = (generated_images_dict['fake_ct'] + 1024) / 4024
        error_image = generated_images_dict['error_ct']

        mr_center_slice_num = mr_images.shape[-1] // 2
        ct_center_slice_num = ct_images.shape[-1] // 2


        visual_s_list += [
            ('real_mr', mr_images[..., mr_center_slice_num].cpu()), 
            ('fake_ct', fake_ct[..., mr_center_slice_num].cpu()), 
            ('error_ct', error_image[..., mr_center_slice_num].cpu()),
            ('real_ct', ct_images[..., mr_center_slice_num].cpu()), 
            ('window_0_100_fake_ct', window_0_100_fake_ct[..., mr_center_slice_num].cpu()),
            ('window_0_100_real_ct', window_0_100_ct_images[..., mr_center_slice_num].cpu()),
            ('window_100_1500_fake_ct', window_100_1500_fake_ct[..., mr_center_slice_num].cpu()),
            ('window_100_1500_real_ct', window_100_1500_ct_images[..., mr_center_slice_num].cpu()),
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