import cv2
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torchvision
import torchvision.tv_tensors

import utils

LABEL_MAPPING = {
    'without_mask' : 0,
    'with_mask' : 1,
    'mask_weared_incorrect' : 2
    }

REVERSE_LABEL_MAPPING = {
    0 : 'without_mask',
    1 : 'with_mask',
    2 : 'mask_weared_incorrect'
}

class Image:

    def __init__(self, img, boxes, labels):

        self.img = img # RGB, H x W x 3
        
        self.box, self.label = boxes, labels # XMIN,YMIN,XMAX,YMAX

    def show(self):

        img_to_show = self.img

        # bounding box color and text
        color = (144,255,144)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX

        for i in zip(self.box, self.label):

            box_coordinates = i[0]

            xmin, ymin = box_coordinates[0], box_coordinates[1]
            xmax, ymax = box_coordinates[2], box_coordinates[3]

            cv2.rectangle(img_to_show, (xmin, ymin), (xmax, ymax), color=color)
            cv2.putText(img_to_show, i[1], (xmax, ymax), color=color, fontFace=fontFace, fontScale=0.3)

        cv2.imshow("image", cv2.cvtColor(img_to_show, cv2.COLOR_RGB2BGR)) # cv2 expects BGR
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_tensor_image(self):

        img_tensor = torchvision.tv_tensors.Image(
            data=self.img,
            dtype=torch.float32 # for normalization later on
        )

        return img_tensor.permute(2,0,1) # returns 3 x H x W
    
    def get_tensor_image_data(self):

        box_tensor = torchvision.tv_tensors.BoundingBoxes(
            data=self.box,
            format=torchvision.tv_tensors.BoundingBoxFormat('XYXY'),
            canvas_size=self.img.shape[0:2],
            dtype=torch.float32
            )
        
        label_tensor = torch.tensor(
            data=[LABEL_MAPPING[i] for i in self.label],
            dtype=torch.int64,
        )

        return box_tensor, label_tensor

############################
### CUSTOM DATASET CLASS ###
############################

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, tensor_images, tensor_boxes, tensor_labels):
        
        self.imgs = tensor_images
        self.boxes = tensor_boxes
        self.labels = tensor_labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        img = self.imgs[index]
        box = self.boxes[index]
        label = self.labels[index]
        n_objects = len(box)

        target = {}
        target['boxes'] = box
        target['labels'] = label
        target['image_id'] = index
        # XMIN, YMIN, XMAX, YMAX
        target['area'] = (box[:,3] - box[:,1]) * (box[:,2] - box[:,0])
        # assume no crowding
        target['iscrowd'] = torch.zeros(n_objects, dtype=torch.int32)
        
        return img, target
