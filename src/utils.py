import cv2
import numpy as np
import random

from ultralytics import YOLO

from is_msgs.image_pb2 import Image

import json


CONFIG = json.load(open('etc/config/config.json'))


def draw_masks(img, results, opacity=0.5):

    overlay = img.copy()

    for result in results:
        for mask, box in zip(result.masks.xy, result.boxes):

            class_id = int(box.cls)
            color = CONFIG['detection']['colors'][str(class_id)]

            points = np.int32([mask])
            
            cv2.fillPoly(overlay, points, color)

    img = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)

    return img

def put_text(img, text, position, color=(0, 0, 255), font=cv2.FONT_HERSHEY_PLAIN, font_scale=1, thickness=2): 
    
    cv2.putText(img, text, position, font, font_scale, color, thickness)

    return img

def draw_boxes(img, results):

    detections = results[0].boxes

    for detection in detections:

        bounding_box = detection.xyxy[0].cpu().numpy().astype(np.int32)
        class_id = int(detection.cls)

        color = CONFIG['detection']['colors'][str(class_id)]
        classe_name = CONFIG['detection']['label'][str(class_id)]

        x1, y1, x2, y2 = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        img = put_text(img, f'{classe_name}', (x1, y1-5), color=color)

    return img

def to_np(input_image):
    if isinstance(input_image, np.ndarray):
        output_image = input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image

def unpack_message(message):
    frame = message.unpack(Image)
    return frame

def load_model(model_path):
    return YOLO(model_path)
