import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from torchvision.tv_tensors import *
from torchvision.utils import *
from torchvision.transforms import v2

##########################
### DATASET PROCESSING ###
##########################

def get_image_data(img_path, xml_path):

    img_data = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    box_data = []
    label_data = []

    tree = ET.parse(xml_path)

    for object in tree.findall('object'):

        box_label = object.find('name').text

        box_coordinates = object.find('bndbox')
        box_coordinates_XYXY = []
        box_coordinates_XYXY.append(int(box_coordinates.find('xmin').text))
        box_coordinates_XYXY.append(int(box_coordinates.find('ymin').text))
        box_coordinates_XYXY.append(int(box_coordinates.find('xmax').text))
        box_coordinates_XYXY.append(int(box_coordinates.find('ymax').text))

        box_data.append(box_coordinates_XYXY)
        label_data.append(box_label)

    return img_data, box_data, label_data

# expects RGB image input
def preprocess(img, boxes):

    # obtain scale factor for x,y
    size = img.shape[0:2]
    x = size[1]
    y = size[0]
    new_x = 512
    new_y = 512
    x_scale = new_x/x
    y_scale = new_y/y

    # image pre-processing steps
    new_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

    new_boxes = []
    for box in boxes:
        new_box = [
            int(box[0]*x_scale),
            int(box[1]*y_scale),
            int(box[2]*x_scale),
            int(box[3]*y_scale)
        ]

        new_boxes.append(new_box)

    return new_img, new_boxes