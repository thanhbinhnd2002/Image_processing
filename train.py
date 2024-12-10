import os.path

import torch
import torch.nn as nn
from click import progressbar
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader
from pprintpp import pprint
from torchvision.transforms import Compose, ToTensor, Resize
from tqdm.autonotebook import tqdm
from torchmetrics.detection import MeanAveragePrecision
import argparse
import shutil


class VOCDataset(VOCDetection):
    def __init__(self, root, year, image_set, download, transform):
        super().__init__(root, year, image_set, download, transform)
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']

    def __getitem__(self, item):
        image, ori_target = super().__getitem__(item)
        boxes = []
        labels = []
        for obj in ori_target["annotation"]["object"]:
            bbox = obj["bndbox"]
            xmin = int(bbox["xmin"])
            ymin = int(bbox["ymin"])
            xmax = int(bbox["xmax"])
            ymax = int(bbox["ymax"])
            label = obj["name"]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes.index(label))
        final_target = {"boxes": torch.FloatTensor(boxes), "labels": torch.LongTensor(labels)}
        return image, final_target

def collate_fn(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets

def get_args():
    parser = argparse.ArgumentParser(description="Train Faster RCNN model for PASCAL VOC dataset")
    parser.add_argument("--data_path", "-d", type=str, default="data", help="Path to the dataset")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=8)
    parser.add_argument("--lr", "-l", type=float, default=1e-3)
    parser.add_argument("--saved_path", "-s", type=str, default="trained_models")
    parser.add_argument("--log_path", "-o", type=str, default="tensorboard")
    parser.add_argument("--checkpoint_path", "-c", type=str, default=None)

    args = parser.parse_args()
    return args


def train(args):
    if type(args) != argparse.Namespace:
        parser = argparse.ArgumentParser(description="Train Faster RCNN model for PASCAL VOC dataset")
        parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
        parser.add_argument("--batch-size", "-b", type=int, default=8)
        parser.add_argument("--lr", "-l", type=float, default=1e-3)
        args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        ToTensor(),
    ])
    train_dataset = VOCDataset(args.data_path, year="2012", image_set="train", download=False, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_dataset = VOCDataset(args.data_path, year="2012", image_set="val", download=False, transform=transform)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
    )
    num_classes = len(train_dataset.classes)
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT, trainable_backbone_layers=6)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=in_features, out_features=num_classes)
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=in_features, out_features=num_classes*4)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        start_epoch = checkpoint["epoch"]
        best_map = checkpoint["best_map"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        start_epoch = 0
        best_map = -1

    # Make a folder containing tensorboard
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    writer = SummaryWriter(args.log_path)

    # Make a folder storing checkpoints
    if not os.path.isdir(args.saved_path):
        os.makedirs(args.saved_path)
    num_iters = len(train_dataloader)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan")
        for iter, (images, targets) in enumerate(progress_bar):
            images = [image.to(device) for image in images]

            # advanced way
            # targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

            # basic way
            target_list = []
            for target in targets:
                target_list.append({
                    "boxes": target["boxes"].to(device),
                    "labels": target["labels"].to(device)
                })
            losses = model(images, target_list)
            total_loss = sum([loss for loss in losses.values()])
            progress_bar.set_description("Epoch {}/{}. Loss: {:0.4f}".format(epoch, args.epochs, total_loss))
            writer.add_scalar("Train/Loss", total_loss, global_step=epoch*num_iters+iter)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        model.eval()
        progress_bar = tqdm(val_dataloader, colour="yellow")
        metric = MeanAveragePrecision(iou_type="bbox")
        for images, targets in progress_bar:
            images = [image.to(device) for image in images]
            with torch.no_grad():
                predictions = model(images)
            cpu_predictions = []
            for prediction in predictions:
                cpu_predictions.append({
                            "boxes": prediction["boxes"].to("cpu"),
                            "labels": prediction["labels"].to("cpu"),
                            "scores": prediction["scores"].to("cpu")
                        })
            metric.update(cpu_predictions, targets)
        map = metric.compute()
        pprint(map)
        writer.add_scalar("Test/mAP", map["map"], epoch)
        writer.add_scalar("Test/mAP_50", map["map_50"], epoch)
        writer.add_scalar("Test/mAP_75", map["map_75"], epoch)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch+1,
            "best_map": best_map
        }
        torch.save(checkpoint, os.path.join(args.saved_path, "last.pt"))
        if map["map"] > best_map:
            best_map = map["map"]
            torch.save(checkpoint, os.path.join(args.saved_path, "best.pt"))


if __name__ == '__main__':
    args = get_args()
    train(args)



