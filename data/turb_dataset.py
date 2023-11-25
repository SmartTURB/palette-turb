import torch.utils.data as data
import torch
import h5py
import numpy as np

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

class TurbInpaintDataset(data.Dataset):
    def __init__(
        self,
        dataset_path,
        dataset_name,
        data_len=-1,
        mask_config={},
        image_size=[64, 64],
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        with h5py.File(self.dataset_path, 'r') as f:
            len_dataset = f[self.dataset_name].len()
        self.data_len = data_len if 0 < data_len < len_dataset else len_dataset
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        assert index < self.data_len, f"Index {index} is out of bounds [0, {self.data_len})"
        ret = {}
        with h5py.File(self.dataset_path, 'r') as f:
            img = f[self.dataset_name][index].astype(np.float32)
            img = np.moveaxis(img, -1, 0)
        img = torch.from_numpy(img)
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = f'{index:0{len(str(self.data_len - 1))}}.jpg'
        return ret

    def __len__(self):
        return self.data_len

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox((64, 64), (32, 32), 10, 5))
        elif self.mask_mode == 'center':
            #h, w = self.image_size
            #mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
            #mask = bbox2mask(self.image_size, (28, 28,  8,  8))
            #mask = bbox2mask(self.image_size, (24, 24, 16, 16))
            #mask = bbox2mask(self.image_size, (20, 20, 24, 24))
            #mask = bbox2mask(self.image_size, (16, 16, 32, 32))
            #mask = bbox2mask(self.image_size, (12, 12, 40, 40))
            #mask = bbox2mask(self.image_size, ( 7,  7, 50, 50))
            #mask = bbox2mask(self.image_size, ( 2,  2, 60, 60))
            mask = bbox2mask(self.image_size, ( 1,  1, 62, 62))
        elif self.mask_mode.startswith('center'):
            lg = int(self.mask_mode[6:])
            i1 = 32 - lg//2
            mask = bbox2mask(self.image_size, (i1, i1, lg, lg))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size, (2, 25), (2, 10))
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size, (3, 10))
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox((64, 64), (32, 32), 10, 5))
            irregular_mask = brush_stroke_mask(self.image_size, (3, 10))
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class TurbUncroppingDataset(data.Dataset):
    def __init__(
        self,
        dataset_path,
        dataset_name,
        data_len=-1,
        mask_config={},
        image_size=[64, 64],
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        with h5py.File(self.dataset_path, 'r') as f:
            len_dataset = f[self.dataset_name].len()
        self.data_len = data_len if 0 < data_len < len_dataset else len_dataset
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        assert index < self.data_len, f"Index {index} is out of bounds [0, {self.data_len})"
        ret = {}
        with h5py.File(self.dataset_path, 'r') as f:
            img = f[self.dataset_name][index].astype(np.float32)
            img = np.moveaxis(img, -1, 0)
        img = torch.from_numpy(img)
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = f'{index:0{len(str(self.data_len - 1))}}.jpg'
        return ret

    def __len__(self):
        return self.data_len

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(
                img_shape=(64, 64), mask_mode=self.mask_mode
            ))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(
                    img_shape=(64, 64), mask_mode='onedirection'
                ))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(
                    img_shape=(64, 64), mask_mode='fourdirection'
                ))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)
