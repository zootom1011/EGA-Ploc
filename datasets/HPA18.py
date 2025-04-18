import torch
import os
from PIL import Image
import pandas as pd
from PIL import UnidentifiedImageError
from torchvision import transforms
import torch.utils.data

from .build import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class HPA18(torch.utils.data.Dataset):

    def __init__(self, cfg, filePath, condition="normal", database="MultiHPA", aug=False):
        self.cfg = cfg
        self.filePath = filePath
        self.aug = aug
        self.im_size = cfg.DATA.CROP_SIZE
        self.locations = cfg.CLASSIFIER.LOCATIONS
        self._get_img_info()


    def __len__(self):
        return len(self._data_info)

    def _get_img_info(self):
        data_file = pd.read_csv(self.filePath, header=0, index_col=0)
        self._data_info = []
        if 'locations' in data_file.columns:
            data_file = data_file[['URL'] + self.locations]
            data_file[self.locations] = data_file[self.locations].astype(int)
            for item in data_file.itertuples(index=True):
                index, url = item[0 : 2]
                locations = item[2 : ]

                im_path = os.path.join(url)

                self._data_info.append({"index": index, "im_path": im_path, "annotations": locations})
        else:
            data_file = data_file[['URL']]
            for item in data_file.itertuples(index=True):
                index, url = item

                im_path = os.path.join(url)

                self._data_info.append(
                    {"index": index, "im_path": im_path, "annotations": []})

    def __load__(self, index):
        im_path = self._data_info[index]["im_path"]
        img = Image.open(im_path).convert('RGB')

        t = []
        if self.aug:
            t.append(transforms.RandomRotation(degrees=(90)))
            t.append(transforms.RandomHorizontalFlip(p=0.5))
            t.append(transforms.RandomVerticalFlip(p=0.5))
        w, h = img.size
        if h < 1000 or w < 1000:
            t.append(transforms.CenterCrop(0.9 * min(w, h)))
            if self.aug:
                t.append(transforms.RandomCrop(min(w, h) * self.im_size // 3000))
            else:
                t.append(transforms.CenterCrop(min(w, h) * self.im_size // 3000))
            t.append(transforms.Resize([self.im_size, self.im_size]))
        else:
            if h < self.im_size or w < self.im_size:
                left_pad = top_pad = right_pad = bottom_pad = 0
                if h < self.im_size:
                    top_pad = (self.im_size + 1 - h) // 2
                    bottom_pad = self.im_size - h - top_pad
                if w < self.im_size:
                    left_pad = (self.im_size + 1 - w) // 2
                    right_pad = self.im_size - w - left_pad
                t.append(transforms.Pad((left_pad, top_pad, right_pad, bottom_pad), padding_mode="reflect"))
            if h > self.im_size or w > self.im_size:
                t.append(transforms.CenterCrop(max(0.9 * min(w, h), self.im_size)))
                if self.aug:
                    t.append(transforms.RandomCrop(self.im_size))
                else:
                    t.append(transforms.CenterCrop(self.im_size))
                # t.append(transforms.Resize([self.im_size, self.im_size]))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(self.cfg.DATA.MEAN, self.cfg.DATA.STD))
        transform = transforms.Compose(t)
        img = transform(img)

        return img

    def __getitem__(self, index):
        idx = self._data_info[index]["index"]
        img = self.__load__(index)
        label = self._data_info[index]["annotations"]
        label = torch.FloatTensor(label)

        return idx, img, label