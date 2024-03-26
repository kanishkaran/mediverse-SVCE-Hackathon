import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, json_file, transform=None, image_size=(640, 640)):
        self.data = json.load(open(json_file))
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, idx):
        image_info = self.data["images"][idx]
        image_path = image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        
        # Resize image
        image = image.resize(self.image_size)

        annotations = []
        for annotation in self.data["annotations"]:
            if annotation["image_id"] == image_info["id"]:
                # Adjust annotation coordinates based on resized image
                bbox = annotation["bbox"]
                bbox = [bbox[0] * (self.image_size[0] / image_info["width"]),
                        bbox[1] * (self.image_size[1] / image_info["height"]),
                        bbox[2] * (self.image_size[0] / image_info["width"]),
                        bbox[3] * (self.image_size[1] / image_info["height"])]
                annotation["bbox"] = bbox
                annotations.append(annotation)

        # Perform any necessary preprocessing
        if self.transform:
            image = self.transform(image)
            
        return image, annotations

# Function to get unique classes from a dataset
def get_unique_classes(dataset):
    unique_classes = set()
    for annotation in dataset.data["annotations"]:
        unique_classes.add(annotation["category_id"])
    return unique_classes

# Paths to JSON files for train, test, and validation subsets
train_json_file = r"D:\code\hackathon\word_recog\Merged.v1i.coco\new_codes\dataset_rname_test\renamed\train\updated_coco.json"
test_json_file = r"D:\code\hackathon\word_recog\Merged.v1i.coco\new_codes\dataset_rname_test\renamed\test\updated_coco.json"
valid_json_file = r"D:\code\hackathon\word_recog\Merged.v1i.coco\new_codes\dataset_rname_test\renamed\valid\updated_coco.json"

# Load datasets
train_dataset = CustomDataset(train_json_file)
test_dataset = CustomDataset(test_json_file)
valid_dataset = CustomDataset(valid_json_file)

# Get unique classes for each dataset
train_classes = get_unique_classes(train_dataset)
test_classes = get_unique_classes(test_dataset)
valid_classes = get_unique_classes(valid_dataset)

# Print unique classes for each subset
print("Unique classes in train dataset:", len(train_classes))
print("Unique classes in test dataset:", len(test_classes))
print("Unique classes in valid dataset:", len(test_classes))
 
num_classes = len(train_classes)+len(test_classes)+len(valid_classes)

# Define Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = len(train_classes)  # Assuming each class has a unique ID
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

# Define optimizer and loss function
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# Define your loss function
# e.g., criterion = ...

def evaluate_model(model, data_loader):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [F.to_tensor(image) for image in images]
            targets = [{k: v for k, v in t.items()} for t in targets]
            predictions = model(images)
            all_predictions.extend(predictions)
            all_targets.extend(targets)

    # Calculate mAP for each class
    num_classes = model.roi_heads.box_predictor.cls_score.out_features
    average_precisions = []

    for class_id in range(num_classes):
        true_positives = []
        false_positives = []
        scores = []

        for predictions, targets in zip(all_predictions, all_targets):
            boxes = predictions['boxes'][predictions]


# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = [F.to_tensor(image) for image in images]
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    # Validation after each epoch
    model.eval()
    mAP = evaluate_model(model, valid_loader)
    print("Mean Average Precision (mAP) on validation dataset after epoch {}: {}".format(epoch+1, mAP))