import os.path
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
import argparse
import cv2
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Faster RCNN's inference for image")
    parser.add_argument("--image_path", "-i", type=str, default="./image.jpg", help="Path to an image")
    parser.add_argument("--checkpoint_path", "-c", type=str, default="trained_models/best.pt")
    parser.add_argument("--conf_threshold", "-t", type=float, default=0.3, help="Confident threshold")

    args = parser.parse_args()
    return args


def train(args):
    print("args: ", args)
    print(type(args))
    if type(args) != argparse.Namespace:
        parser = argparse.ArgumentParser(description="Faster RCNN's inference for image")
        file_path = args["file_path"]
        conf_threshold = args["conf_threshold"]
        parser.add_argument("--image_path", "-i", type=str, default=f"{file_path}", help="Path to an image")
        parser.add_argument("--checkpoint_path", "-c", type=str, default="trained_models/best.pt")
        parser.add_argument("--conf_threshold", "-t", type=float, default=f"{conf_threshold}", help="Confident threshold")

        args = parser.parse_args()

    classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
    num_classes = len(classes)

    # Tạo mô hình
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT, trainable_backbone_layers=6)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=in_features, out_features=num_classes)
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=in_features, out_features=num_classes*4)

    # Tải checkpoint từ GPU về CPU
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))  # Chuyển checkpoint về CPU
        model.load_state_dict(checkpoint["model"])
    else:
        print("Checkpoint không tồn tại hoặc không hợp lệ.")
        exit(0)

    model.eval()

    # Đọc và chuẩn bị ảnh
    ori_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB) / 255.0
    image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()

    # Dự đoán trên ảnh
    with torch.no_grad():
        predictions = model([image])[0]
    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if score > args.conf_threshold:
            xmin, ymin, xmax, ymax = box.int().tolist()
            cv2.rectangle(ori_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(ori_image, classes[label], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 0), 4, cv2.LINE_AA)

    # Lưu ảnh kết quả
    cv2.imwrite("result.jpg", ori_image)
    print("Kết quả đã được lưu tại: result.jpg")
    return "result.jpg"






if __name__ == '__main__':
    args = get_args()
    train(args)



