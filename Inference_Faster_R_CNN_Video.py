import os.path

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
import argparse
import cv2
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Faster RCNN's inference for video")
    parser.add_argument("--video_path", "-v", type=str, default="./test_2.mp4", help="Path to a video")
    parser.add_argument("--out_path", "-o", type=str, default="./result.mp4", help="Path to output video")
    parser.add_argument("--checkpoint_path", "-c", type=str, default="trained_models/best.pt")
    parser.add_argument("--conf_threshold", "-t", type=float, default=0.5, help="Confident threshold")

    args = parser.parse_args()
    return args


def train(args):
    classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
    num_classes = len(classes)
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT, trainable_backbone_layers=6)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=in_features, out_features=num_classes)
    model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=in_features, out_features=num_classes*4)

    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))  # Chuyển checkpoint về CPU
        model.load_state_dict(checkpoint["model"])
    else:
        exit(0)
    model.eval()

    cap = cv2.VideoCapture(args.video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    writer = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc(*"mp4"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (width, height))
    while cap.isOpened():
        flag, ori_frame = cap.read()
        if not flag:
            break

        frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB)/255.
        frame = torch.from_numpy(np.transpose(frame, (2, 0, 1))).float()
        with torch.no_grad():
            predictions = model([frame])[0]
        for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
            if score > args.conf_threshold:
                xmin, ymin, xmax, ymax = box.int().tolist()
                cv2.rectangle(ori_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                cv2.putText(ori_frame, classes[label], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 255, 0), 4, cv2.LINE_AA)
        writer.write(ori_frame)

    cap.release()
    writer.release()






if __name__ == '__main__':
    args = get_args()
    train(args)



