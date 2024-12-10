import torch
import torchvision
import myOwnDataset
from Prototype.myOwnDataset import CustomCocoDataset

train_data_dir = "C:/Users/pawel/Documents/Houses"
train_coco = "C:/Users/pawel/Documents/Houses/instances_default.json"

# In my case, just added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

# create own Dataset
my_dataset = CustomCocoDataset(root=train_data_dir,
                          annotation=train_coco,
                          transforms=get_transform()
                          )

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

# Batch size
train_batch_size = 1

# own DataLoader
data_loader = torch.utils.data.DataLoader(my_dataset,
                                          batch_size=train_batch_size,
                                          shuffle=True,
                                          num_workers=4,
                                          collate_fn=collate_fn)