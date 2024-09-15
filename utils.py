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

def get_image_data(path_to_xml):

    box_data = []
    label_data = []

    tree = ET.parse(path_to_xml)

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

    return box_data, label_data

# expects RGB image input
def image_preprocess(IMAGE):

    img = IMAGE.img
    box_data = IMAGE.box

    # obtain scale factor for x,y
    size = IMAGE.size()
    x = size[1]
    y = size[0]
    new_x = 512
    new_y = 512
    x_scale = new_x/x
    y_scale = new_y/y

    # image pre-processing steps
    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    new_img = cv2.resize(new_img, (512, 512), interpolation=cv2.INTER_LINEAR)

    new_box = {}
    for label in box_data:

        box_coordinates = box_data[label]

        new_b = [
            int(box_coordinates[0]*x_scale),
            int(box_coordinates[1]*y_scale),
            int(box_coordinates[2]*x_scale),
            int(box_coordinates[3]*y_scale)
        ]

        new_box[label] = new_b

    return new_img, new_box