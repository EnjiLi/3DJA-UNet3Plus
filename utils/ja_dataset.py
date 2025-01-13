"""
UNDER LPG LICENSE
3D Joint Attention for Semantic Segmentation daataset

@Author: Ibra Ndiaye
@Dat: 2025-01-13
"""

import os
from PIL import Image
import numpy as np
import torch
class ThreeDJaDataset(torch.utils.data.Dataset):
	def __init__(self, root: str, train: bool, transforms=None):
		super(ThreeDJaDataset, self).__init__()
		self.flag = "train" if train else "test"
		data_root = os.path.join(root,  self.flag)
		assert os.path.exists(data_root), f"path '{data_root}' does not exists."
		self.transforms = transforms
		img_names = [i for i in os.listdir(os.path.join(data_root, "image")) if i.endswith(".tif")]
		self.img_list = [os.path.join(data_root, "image", i) for i in img_names]
		self.manual = [os.path.join(data_root, "label", i) for i in img_names]
		# check files
		for i in self.manual:
			if os.path.exists(i) is False:
				raise FileNotFoundError(f"file {i} does not exists.")
	
	def __getitem__(self, idx):
		img = Image.open(self.img_list[idx]).convert('RGB')
		manual = Image.open(self.manual[idx]).convert('L')
		manual = np.array(manual) / 255
		mask = np.clip(manual, a_min=0, a_max=255)
		
		# The reason for switching back to PIL here is that transforms processes PIL data.
		mask = Image.fromarray(mask)
		
		if self.transforms is not None:
			img, mask = self.transforms(img, mask)
		
		return img, mask
	
	def __len__(self):
		return len(self.img_list)
	

	def collate_fn(self, batch):
		images, targets = list(zip(*batch))
		batched_imgs = self.cat_list(images, fill_value=0)
		batched_targets = self.cat_list(targets, fill_value=255)
		return batched_imgs, batched_targets

	@staticmethod
	def cat_list(images, fill_value=0):
		max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
		batch_shape = (len(images),) + max_size
		batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
		for img, pad_img in zip(images, batched_imgs):
			pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
		return batched_imgs
