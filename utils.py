import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from torchvision.tv_tensors import *
from torchvision.utils import *

def data_to_tensor(img_path, box_path):

    toReturn = {}
    
    img = cv2.imread(img_path) # BGR, H x W x C
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = Image(
        np.moveaxis(img_rgb, source=[0,1,2], destination=[1,2,0])
    )
    img_shape = img_tensor.size() # RGB, C x H x W
    
    tree = ET.parse(box_path)
    all_box_coordinates_XYXY = []
    labels = []

    for object in tree.findall('object'):

        box_name = object.find('name').text
        labels.append(box_name)

        box_coordinates = object.find('bndbox')
        box_coordinates_XYXY = []
        box_coordinates_XYXY.append(int(box_coordinates.find('xmin').text))
        box_coordinates_XYXY.append(int(box_coordinates.find('ymin').text))
        box_coordinates_XYXY.append(int(box_coordinates.find('xmax').text))
        box_coordinates_XYXY.append(int(box_coordinates.find('ymax').text))

        all_box_coordinates_XYXY.append(box_coordinates_XYXY)
    
    box_tensor = BoundingBoxes(
        all_box_coordinates_XYXY,
        format = BoundingBoxFormat.XYXY,
        canvas_size = img_shape[:-1]
    )

    toReturn['img'] = img_tensor
    toReturn['box'] = box_tensor
    toReturn['labels'] = labels

    return toReturn


###############
### TESTING ###
###############
"""
test_img = "DATASET/IMAGES/maksssksksss0.png"
test_box = "DATASET/ANNOTATIONS/maksssksksss0.xml"

test_data = data_to_tensor(test_img, test_box)
print(test_data)

test_plot = draw_bounding_boxes(test_data['img'], test_data['box'],
                                labels=test_data['labels'], colors=['green', 'green', 'green'])
plt.imshow(torch.moveaxis(test_plot, 0, 2))
plt.show()
"""
