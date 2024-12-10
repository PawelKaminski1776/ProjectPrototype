import os
import sys
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import timeit

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define label mapping
CATEGORY_NAMES = {
    1: "Lawn",
    2: "Lights",
    3: "Car",
    4: "Bins",
    5: "Human"
}

def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

def visualize_combined_predictions(image_path, all_predictions):
    image = Image.open(image_path)
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    ax = plt.gca()

    colors = ["r", "g", "b", "y", "m", "c"]  # Colors for different epochs
    for epoch_idx, predictions in enumerate(all_predictions):
        color = colors[epoch_idx % len(colors)]  # Cycle through colors
        for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
            if score > 0.3:  # Only display predictions with score > 0.3
                x1, y1, x2, y2 = box.cpu().numpy()
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                category_name = CATEGORY_NAMES.get(label.item(), f"Label {label.item()}")
                ax.text(x1, y1, f"{category_name} ({score:.2f})", color='white',
                        bbox=dict(facecolor=color, alpha=0.5))

    plt.title("Combined Predictions Across All Epochs (Score > 0.3)")
    plt.axis('off')
    plt.show()




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    num_classes = 6
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    checkpoint_dir = "./Model"
    checkpoint_files = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("fasterrcnn_model_epoch")])

    if not checkpoint_files:
        print("No checkpoints found in the directory.")
        sys.exit(1)

    all_predictions = []

    for checkpoint_file in checkpoint_files:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        print(f"Loading checkpoint: {checkpoint_file}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()

        try:
            input_image = preprocess_image(image_path).to(device)
            input_image = [input_image]

            with torch.no_grad():
                predictions = model(input_image)
                all_predictions.append(predictions)

            print(f"Predictions for {checkpoint_file} (Score > 0.3):")
            for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
                if score > 0.3:  # Only print predictions with score > 0.3
                    category_name = CATEGORY_NAMES.get(label.item(), f"Label {label.item()}")
                    print(f"  {category_name}: {score:.2f} (Box: {box.cpu().numpy()})")
        except Exception as e:
            print(f"Error processing the image for checkpoint {checkpoint_file}: {e}")

    visualize_combined_predictions(image_path, all_predictions)
