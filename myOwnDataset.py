import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO


class CustomCocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]

        # Load annotations for the given image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)

        # Load image
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")

        # Prepare bounding boxes
        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for annotation in annotations:
            xmin, ymin, width, height = annotation['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(annotation['category_id'])
            areas.append(annotation['area'])
            iscrowd.append(annotation['iscrowd'])

        # Convert lists to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        img_id_tensor = torch.tensor([img_id])

        # Create annotation dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": img_id_tensor,
            "area": areas,
            "iscrowd": iscrowd
        }

        # Apply transformations
        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        """
        Get the length of the dataset.

        :return: Total number of items in the dataset.
        """
        return len(self.ids)
