import torch
import numpy as np
from torchvision.utils import make_grid, save_image
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
from tensorboardX import SummaryWriter
from PIL import Image


def visualize_multi_slice(input_tensor: torch.FloatTensor, save_path):
    if input_tensor.shape[1] == 2:
        input_tensor = input_tensor[:, 0, :, :, :]

    if type(input_tensor) == np.ndarray:
        input_tensor = torch.from_numpy(input_tensor)

    if input_tensor.ndim >= 3:
        input_tensor = torch.squeeze(input_tensor)

    slice_count = input_tensor.shape[0]

    # shift the value range to [0, 1]
    input_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min())
    input_tensor = input_tensor.unsqueeze(dim=1)  # shape of (slice_count, 1, H, W)
    # print(input_tensor.shape)
    grid_tensors = make_grid(input_tensor, nrow=4, padding=2)
    save_image(grid_tensors, save_path)


def visualize_img_seg_overlay(image_path, segmentation_path, output_path):
    # Load the image and segmentation data
    image_data = nib.load(image_path).get_fdata()
    segmentation_data = nib.load(segmentation_path).get_fdata()
    segmentation_data[segmentation_data != 0] = 1

    # Normalize image data to [0, 1] range
    image_data = np.clip(image_data, -50, 150)
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    # Calculate an appropriate number of rows and columns for the grid
    num_slices = image_data.shape[2]
    num_rows = 4  # You can adjust this number to your preference
    num_cols = (num_slices + num_rows - 1) // num_rows

    # Calculate figsize to maintain aspect ratio
    aspect_ratio = image_data.shape[0] / image_data.shape[1]
    fig_width = 15
    fig_height = fig_width * aspect_ratio * num_rows / num_cols
    figsize = (fig_width, fig_height)

    # Set up the figure and subplots
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=(num_rows, num_cols), axes_pad=0.1)

    # Iterate through axial slices and visualize
    for i, ax in enumerate(grid):
        if i < num_slices:
            ax.imshow(image_data[:, :, i], cmap='gray')
            ax.imshow(segmentation_data[:, :, i], cmap='jet', alpha=0.5)

            ax.axis('off')
            
            plt.cla()

    # Save the figure as a PNG file
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.figure().clear()
    plt.close('all')

    print(f"Visualization saved as {output_path}")


def visualize_img(image_path, output_path):
    # Load the image and segmentation data
    if isinstance(image_path, str):
        image_data = nib.load(image_path).get_fdata()
    elif isinstance(image_path, np.ndarray):
        image_data = image_path
    else:
        raise Exception("Unexpected type of image_path")

    # Normalize image data to [0, 1] range
    image_clip_low, image_clip_high = np.percentile(image_data, [0.5, 99.5])
    image_data = np.clip(image_data, image_clip_low, image_clip_high)
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

    # Calculate an appropriate number of rows and columns for the grid
    num_slices = image_data.shape[2]
    num_rows = 4  # You can adjust this number to your preference
    num_cols = (num_slices + num_rows - 1) // num_rows

    # Calculate figsize to maintain aspect ratio
    aspect_ratio = image_data.shape[0] / image_data.shape[1]
    fig_width = 15
    fig_height = fig_width * aspect_ratio * num_rows / num_cols
    figsize = (fig_width, fig_height)

    # Set up the figure and subplots
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=(num_rows, num_cols), axes_pad=0.1)

    # Iterate through axial slices and visualize
    for i, ax in enumerate(grid):
        if i < num_slices:
            ax.imshow(image_data[:, :, i], cmap='gray')

            ax.axis('off')
            
            plt.cla()

    # Save the figure as a PNG file
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    
    plt.figure().clear()
    plt.close('all')

    print(f"Visualization saved as {output_path}")


class Visualizer(object):
    def __init__(self, config, out_dir):
        self.config = config
        
        self.log_dir = os.path.join(out_dir, 'log')
        self.GEN_IMG_DIR = os.path.join(out_dir, 'generated_imgs')
        os.makedirs(self.GEN_IMG_DIR, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        self.valid_colors = [
            [  0,   0,  0],
            [254, 232, 81], # yellow LV-myo
            [145, 193, 62], # green LA-blood
            [ 29, 162, 220], # blue LV-blood
            [238,  37,  36]]  # Red AA
            
        self.label_colours = dict(zip(range(5), self.valid_colors))
    
    def decode_segmap(self, img):  # img is numpy.array object
        map = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3))
        for idx in range(img.shape[0]):
            temp = img[idx, :, :]
            r = temp.copy()
            g = temp.copy()
            b = temp.copy()
            for l in range(0, 5):
                r[temp == l] = self.label_colours[l][0]
                g[temp == l] = self.label_colours[l][1]
                b[temp == l] = self.label_colours[l][2]

            rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
            rgb[:, :, 0] = r / 255.0
            rgb[:, :, 1] = g / 255.0
            rgb[:, :, 2] = b / 255.0
            map[idx, :, :, :] = rgb
        return map

    def display_current_results(self, visuals, domain, step, mode='train'):
        imgs = []
        for key, t in visuals.items():
            # resize the visulas to 256x256 image
            # t_resize = F.interpolate(t, (256, 256), mode='bicubic')
            if 'seg' in key:
                img_seg = self.decode_segmap(t)
                imgs.append(torch.tensor(img_seg.transpose((0, 3, 1, 2)), dtype=torch.float))
            else:
                # if t.min() >= 0 and t.max() <= 1:
                #     pass
                # else:
                #     t = (t + 1) / 2
                imgs.append(t.expand(-1, 3, -1, -1).cpu())
        
        imgs = torch.cat(imgs, 0)     #Concatenates the given sequence of seq tensors in the given dimension.
        imgs = make_grid(imgs.detach(), nrow=self.config['batch_size'], normalize=False, scale_each=False).cpu().numpy()   #Make a grid of images.
        imgs = np.clip(imgs * 255, 0, 255).astype(np.uint8)   #限制数组值在一定范围 若小于0 则变为0
        imgs = imgs.transpose((1, 2, 0))
        imgs = Image.fromarray(imgs)
        filename = '%05d_%s_%s.jpg' % (step, mode, "source" if domain=="s" else "target")
        imgs.save(os.path.join(self.GEN_IMG_DIR, filename))

    def plot_current_errors(self, errors, step):
        
        for tag, value in errors.items():
            if tag == "cur_evaluate_dice":
                pass
            else:
                value = value.mean().cpu().numpy()
            self.writer.add_scalar(tag, value, step)

    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %s) ' % (epoch, i, t)
        for k, v in errors.items():
            #print(v)
            #if v != 0:
            v = v.mean().numpy()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
