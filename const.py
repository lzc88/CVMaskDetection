import cv2
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torchvision
import torchvision.tv_tensors

import utils

LABEL_MAPPING = {
    'without_mask' : 0,
    'with_mask' : 1
    }

class Image:

    def __init__(self, img_path, box_path):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # RGB, H x W x C
        self.img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        # XMIN,YMIN,XMAX,YMAX
        self.box, self.label = utils.get_image_data(box_path)

        # H x W
        self.size = self.img.shape[0:2]

    def show(self):

        img_to_show = self.img

        # bounding box color and text
        color = (144,255,144)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX

        for label in self.box:

            box_coordinates = self.box[label]

            xmin, ymin = box_coordinates[0], box_coordinates[1]
            xmax, ymax = box_coordinates[2], box_coordinates[3]

            cv2.rectangle(img_to_show, (xmin, ymin), (xmax, ymax), color=color)
            cv2.putText(img_to_show, label, (xmax, ymax), color=color, fontFace=fontFace, fontScale=0.3)

        cv2.imshow("image", cv2.cvtColor(img_to_show, cv2.COLOR_RGB2BGR)) # cv2 expects BGR
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_tensor_image(self):

        img_tensor = torchvision.tv_tensors.Image(
            data=self.img,
            dtype=torch.float64, # for normalization later on
            device=torch.device(self.device)
        )

        return img_tensor
    
    def get_tensor_image_data(self):

        box_tensor = torchvision.tv_tensors.BoundingBoxes(
            data=self.box,
            format=torchvision.tv_tensors.BoundingBoxFormat('XYXY'),
            canvas_size=self.size,
            dtype=torch.int64,
            device=self.device
        )

        label_tensor = torch.tensor(
            data=[LABEL_MAPPING[i] for i in self.label],
            dtype=torch.int64,
            device=self.device
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
        target['iscrowd'] = torch.zeros(n_objects, dtype=torch.int64)
        
        return img, target
    
    def __len__(self):
        return len(self.imgs)
