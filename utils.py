import math
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)
import cv2
import numpy as np
import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

mask = np.zeros(len(CLASSES),dtype ='uint8')
req_class_label  = [2,3,4,6,7,8] 
for label in req_class_label:
  mask[label] = 1
mask = torch.tensor(mask).to(device)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b

def show(pil_img, prob, boxes,fps):
    colors = COLORS * 100
    np_img = np.array(pil_img)
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            cv2.rectangle(np_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), c, 3)
            cl = p.argmax()
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
            cv2.putText(np_img, text, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX,0.7,tuple(c), 2)
            cv2.putText(np_img, f"fps: {fps:.2f}", (10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 2)
    cv2.imshow('frames', np_img)