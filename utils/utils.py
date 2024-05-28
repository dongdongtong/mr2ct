import torch
import numpy as np
import random


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)
        

def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


def get_image_slicer_to_crop(nonzero_mask):
    outside_value = 0
    mask_voxel_coords = np.where(nonzero_mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[2]))
    maxzidx = int(np.max(mask_voxel_coords[2])) + 1
    minxidx = int(np.min(mask_voxel_coords[3]))
    maxxidx = int(np.max(mask_voxel_coords[3])) + 1
    minyidx = int(np.min(mask_voxel_coords[4]))
    maxyidx = int(np.max(mask_voxel_coords[4])) + 1
    bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]
    return bbox


def process_valid_torch_tensor(valid_img_tensor: torch.FloatTensor, valid_seg_tensor: torch.FloatTensor, img_size=(24, 320, 320), z_index=0):
    """For valid_tensor layout of NCDHW = (1, 1, D, H, w), we first crop the valid_tensor then normalize it
    """
    
    # torch_mask_coords = torch.where(valid_seg_tensor != 0)
    # minzidx = int(torch.min(torch_mask_coords[0]))
    # maxzidx = int(torch.max(torch_mask_coords[0]))+1
    # minxidx= int(torch.min(torch_mask_coords[1]))
    # maxxidx = int(torch.max(torch_mask_coords[1])) + 11
    # minyidx = int(torch.min(torch_mask_coords[2]))
    # maxyidx = int(torch.max(torch_mask_coords[2])) + 1
    # seg_bbox = [[minzidx,maxzidx], [minxidx, maxxidx], [minyidx, maxyidx],]
    seg_bbox = get_image_slicer_to_crop(valid_seg_tensor.numpy())
    
    n, c, d, h, w = valid_seg_tensor.shape
    img_z_center = d // 2
    target_z_shape = img_size[z_index]
    
    # init center crop z
    init_crop_start = img_z_center - target_z_shape // 2
    init_crop_end = init_crop_start + target_z_shape
    # print(init_crop_start, init_crop_end)
    
    lesion_start = seg_bbox[z_index][0]
    lesion_end = seg_bbox[z_index][1]
    
    target_z_start, target_z_end = 0, 0
    
    if init_crop_start <= lesion_start and init_crop_end >= lesion_end:  # which means the center crop just contains the lesion slices
        target_z_start = init_crop_start
        target_z_end = init_crop_end
    
    if lesion_start < init_crop_start and lesion_end > init_crop_end:  # which means the lesion slices is too many
        lesion_center = (lesion_start + lesion_end) // 2
        crop_start = lesion_center - target_z_shape // 2
        crop_end = crop_start + target_z_shape // 2
        
        target_z_start = crop_start
        target_z_end = crop_end
    
    if lesion_start < init_crop_start:
        crop_start = lesion_start
        crop_end = crop_start + target_z_shape
        
        target_z_start = crop_start
        target_z_end = crop_end
    
    if init_crop_end < lesion_end:
        crop_end = lesion_end
        crop_start = crop_end - target_z_shape
        
        target_z_start = crop_start
        target_z_end = crop_end
    
    # print(valid_img_tensor.shape, valid_seg_tensor.shape, seg_bbox, lesion_start, lesion_end, target_z_start, target_z_end)
    
    target_img_tensor = valid_img_tensor[:, :, target_z_start:target_z_end, :, :]
    target_img_tensor = torch.clamp(target_img_tensor, 0, 100) / 100
    target_seg_tensor = valid_seg_tensor / 2
    target_seg_tensor = target_seg_tensor[:, :, target_z_start:target_z_end, :, :]
    
    return target_img_tensor, target_seg_tensor