import re
import os
import cv2
import torch
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import my_utils as utils
import math
import sys
from my_engine import evaluate


directory = "PennFudanPed/Annotation/"


def parse_pascal(pascal_txt_path, convert2OpenCV=False):
    object_bboxes = []  # Store each bbox as [Xmin, Ymin, Xmax, Ymax]
    with open(pascal_txt_path) as file:
        lines = file.readlines()
        start_str = "Bounding box"
        for line in lines:
            if line[:len(start_str)] == start_str:
                coordinates = list(map(int, re.findall(r'\d+', line)))[1:]
                # coordinates uses the Pacal 1.00 format [Xmin, Ymin, Xmax, Ymax]
                if convert2OpenCV:
                    object_bboxes.append([coordinates[0], coordinates[1],  # Xmin, Ymin (top left corner)
                                          coordinates[2] - coordinates[0],  # width
                                          coordinates[3] - coordinates[1]  # height
                                          ])
                else:
                    object_bboxes.append(coordinates)
    return object_bboxes


def image_shape_HWC2CHW(img):
  # Changes img shape from [H, W, C] -> [C, H, W]
  return np.einsum('kli -> ilk', img)


def image_shape_CHW2HWC(img):
  # Changes img shape from [H, W, C] -> [C, H, W]
  return np.einsum('ilk -> kli', img)


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.bboxes = list(sorted(os.listdir(os.path.join(root, "Annotation"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        bboxes_path = os.path.join(self.root, "Annotation", self.bboxes[idx])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # OpenCV
        img = image_shape_HWC2CHW(img)
        # img = np.einsum('kli -> ilk', img)  # Changes img shape from [H, W, C] -> [C, H, W]
        img = torch.from_numpy(img).float()

        # get bounding box coordinates for pedestrian
        bboxes = parse_pascal(bboxes_path, convert2OpenCV=False)
        num_objs = len(bboxes)

        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
      return len(self.imgs)


def collate_fn(batch):
    return tuple(zip(*batch))

def get_instance_detector_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return metric_logger


def main():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # use our dataset and defined transformations
    dataset = PennFudanDataset('PennFudanPed')  # get_transform(train=True)
    dataset_test = PennFudanDataset('PennFudanPed') # get_transform(train=False)

    targets = []
    images = []
    for image in range(len(dataset)):
        targets.append(dataset[image][1])
        images.append(dataset[image][0])


    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # Change batch size to prevent RAM OOM (attempted: 10, 5)
    b_size = 2
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=b_size, shuffle=True, num_workers=1,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=b_size, shuffle=False, num_workers=1,
        collate_fn=collate_fn)
    # TODO: Find an optimal batch size, batch sizes non-divisible by dataset results
    # in error during train_one_epoch fxn. batch_size=10 works, but results in OOM error

    images, targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = model(images, targets)   # Returns losses and detections
    # For inference
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)  # Returns predictions

    ###

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2

    # get the model using our helper function
    model = get_instance_detector_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.00005,
                                momentum=0.9, weight_decay=0.0005)

    # let's train it for 10 epochs
    from torch.optim.lr_scheduler import StepLR
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        # lr_scheduler.step()

    # evaluate on the test dataset
    for epoch in range(num_epochs):
        evaluate(model, data_loader_test, device=device)


if __name__ == "__main__":
    main()
