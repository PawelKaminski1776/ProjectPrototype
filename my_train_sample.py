import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data_loader import data_loader
import torch
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

if __name__ == '__main__':
    num_classes = 6
    num_epochs = 10
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    len_dataloader = len(data_loader)

    os.makedirs("./Model", exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (imgs, annotations) in enumerate(data_loader, 1):
            imgs = [img.to(device) for img in imgs]
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]

            try:
                loss_dict = model(imgs, annotations)
                losses = sum(loss for loss in loss_dict.values())
            except Exception as e:
                print(f"Error during forward pass: {e}")
                continue

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            print(f'Iteration: {i}/{len_dataloader}, Loss: {losses.item()}')

        print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {epoch_loss}")
        torch.save(model.state_dict(), f"./Model/fasterrcnn_model_epoch_{epoch + 1}.pth")
        print(f"Model saved for epoch {epoch + 1}")
