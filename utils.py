import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from torchvision.tv_tensors import *
from torchvision.utils import *
from torchvision.transforms import v2

def get_box_data(path_to_xml):

    box_coordinates_XYXY_all = []
    box_labels = []

    tree = ET.parse(path_to_xml)

    for object in tree.findall('object'):

        box_label = object.find('name').text
        box_labels.append(box_label)

        box_coordinates = object.find('bndbox')
        box_coordinates_XYXY = []
        box_coordinates_XYXY.append(int(box_coordinates.find('xmin').text))
        box_coordinates_XYXY.append(int(box_coordinates.find('ymin').text))
        box_coordinates_XYXY.append(int(box_coordinates.find('xmax').text))
        box_coordinates_XYXY.append(int(box_coordinates.find('ymax').text))

        box_coordinates_XYXY_all.append(box_coordinates_XYXY)

    return box_coordinates_XYXY_all, box_labels

def process_image(IMAGE): # input image RGB

    img = IMAGE.img
    box = IMAGE.box

    size = IMAGE.size()
    x = size[1]
    y = size[0]
    new_x = 512
    new_y = 512
    x_scale = new_x/x
    y_scale = new_y/y

    new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    new_img = cv2.resize(new_img, (512, 512), interpolation=cv2.INTER_LINEAR)
    new_img = cv2.equalizeHist(new_img)

    new_box = []
    for b in box:
        new_b = [b[0]*x_scale, b[1]*y_scale, b[2]*x_scale, b[3]*y_scale]
        new_b = [int(x) for x in new_b]
        new_box.append(new_b)

    return new_img, new_box


def data_to_tensor(img_path, box_path):

    toReturn = {}
    
    img = cv2.imread(img_path) # BGR, H x W x C
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = Image(
        np.moveaxis(img_rgb, source=[0,1,2], destination=[1,2,0])
    )
    img_shape = img_tensor.size() # RGB, C x H x W
    
    box_data = get_box_data(box_path)
    all_box_coordinates_XYXY = box_data['box']
    labels = box_data['labels']
    
    box_tensor = BoundingBoxes(
        all_box_coordinates_XYXY,
        format = BoundingBoxFormat.XYXY,
        canvas_size = img_shape[1::]
    )

    toReturn['img'] = img_tensor
    toReturn['box'] = box_tensor
    toReturn['labels'] = labels

    return toReturn

def transform_tensor(tensor_data):

    def get_img_mean(tensor_img):

        img_mean = torch.mean(transformed['img'], dim=[1,2])

        return img_mean.numpy()
    
    def get_img_std(tensor_img):

        img_std = torch.std(transformed['img'], dim=[1,2])

        return img_std.numpy()

    img = tensor_data.img
    box = tensor_data.box
    
    transforms = v2.Compose([
        v2.Resize(size=(512,512), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=get_img_mean(img), )

    ])

    transformed = transforms({'img':img, 'box':box})

    return transformed