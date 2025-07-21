"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import torch
import torch.utils.data

import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision import datapoints

from pycocotools import mask as coco_mask

from src.core import register

import os
from PIL import Image
from pycocotools.coco import COCO

import random

__all__ = ['CocoDetection']


@register
class CocoDetection(torchvision.datasets.CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']
    
    def __init__(self, img_folder_rgb, img_folder_ir, ann_file_rgb, ann_file_ir, transforms, return_masks, remap_mscoco_category=False):
        super(CocoDetection, self).__init__(img_folder_rgb, ann_file_rgb)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        # rgb
        self.img_folder_rgb = img_folder_rgb
        self.img_folder_ir = img_folder_ir
        # ir
        self.ann_file_rgb = ann_file_rgb
        self.ann_file_ir = ann_file_ir       
        
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

        # Load RGB annotations
        self.coco_rgb = COCO(ann_file_rgb)
        # Load IR annotations
        self.coco_ir = COCO(ann_file_ir)

    def __getitem__(self, idx):
        # Load RGB data
        img_rgb, target_rgb = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target_rgb = {'image_id': image_id, 'annotations': target_rgb}
        img_rgb, target_rgb = self.prepare(img_rgb, target_rgb)

        if 'boxes' in target_rgb:
            target_rgb['boxes'] = datapoints.BoundingBox(
                target_rgb['boxes'],
                format=datapoints.BoundingBoxFormat.XYXY,
                spatial_size=img_rgb.size[::-1]
            )

        if 'masks' in target_rgb:
            target_rgb['masks'] = datapoints.Mask(target_rgb['masks'])

        # Load IR data
        img_id_ir = self.coco_ir.getImgIds()[idx]
        img_info_ir = self.coco_ir.loadImgs(img_id_ir)[0]
        img_path = os.path.join(self.img_folder_ir, img_info_ir['file_name'])
        img_ir = Image.open(img_path).convert('RGB')
        ann_ids = self.coco_ir.getAnnIds(imgIds=img_info_ir['id'])
        target_ir = self.coco_ir.loadAnns(ann_ids)
        target_ir = {'image_id': img_info_ir['id'], 'annotations': target_ir}
        img_ir, target_ir = self.prepare(img_ir, target_ir)

        if 'boxes' in target_ir:
            target_ir['boxes'] = datapoints.BoundingBox(
                target_ir['boxes'],
                format=datapoints.BoundingBoxFormat.XYXY,
                spatial_size=img_ir.size[::-1]
            )

        if 'masks' in target_ir:
            target_ir['masks'] = datapoints.Mask(target_ir['masks'])


        seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(seed)
        random.seed(seed)


        if self._transforms is not None:
            img_rgb, target_rgb = self._transforms(img_rgb, target_rgb)
            torch.manual_seed(seed) 
            random.seed(seed)
            img_ir, target_ir = self._transforms(img_ir, target_ir)

        return torch.cat([img_rgb, img_ir], dim=0), target_rgb


    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'

        return s 


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.remap_mscoco_category:
            classes = [mscoco_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]
            
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])
    
        return image, target


mscoco_category2name = {
    1: 'person',
    2: 'car',
    3: 'bicycle'
}

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}