import torch
import torchvision
from data_loader import data_loader

# select device (whether GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':
    # DataLoader is iterable over Dataset
    for imgs, annotations in data_loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        print(annotations)
