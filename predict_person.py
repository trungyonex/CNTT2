from yolo_model import yolo_utils
import torch
from torchvision import transforms
import numpy as np
import cv2

def get_bbox(yolo_model, device, image_context, yolo_image_size=416, conf_thresh=0.8, nms_thresh=0.4):
    test_transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
    image_yolo = test_transform(cv2.resize(image_context, (416, 416))).unsqueeze(0).to(device)

    with torch.no_grad():
        detections = yolo_model(image_yolo)
        nms_det  = yolo_utils.non_max_suppression(detections, conf_thresh, nms_thresh)[0]
        det = yolo_utils.rescale_boxes(nms_det, yolo_image_size, (image_context.shape[:2]))
  
    bboxes = []
    for x1, y1, x2, y2, _, _, cls_pred in det:
        if cls_pred == 0:  # checking if predicted_class = persons. 
            x1 = int(min(image_context.shape[1], max(0, x1)))
            x2 = int(min(image_context.shape[1], max(x1, x2)))
            y1 = int(min(image_context.shape[0], max(15, y1)))
            y2 = int(min(image_context.shape[0], max(y1, y2)))
            bboxes.append([x1, y1, x2, y2])
    return np.array(bboxes)

def predict_person(path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yolo = yolo_utils.prepare_yolo('yolo_model')
    yolo = yolo.to(device)
    yolo.eval()
    image_context = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    bbox_yolo = get_bbox(yolo, device, image_context)
    return bbox_yolo