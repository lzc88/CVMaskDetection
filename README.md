# Personal CV Project: Mask Detection and Classification

>**NOTE**

- OpenCV library reads and returns images with shape **H x W x 3 (BGR)**

- matplotlib.pyplot expects images with shape **H x W x 3 (RGB)**

- PyTorch represents images as tensors with shape **3 x H x W**

## Dataset

- Using the [Face Mask Detection dataset from Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection), which has 853 images with **3 classes for prediction ( wearing mask, not wearing mask, mask worn incorrectly )**

- Custom `Image` class to load images along with their respective annotations (bounding boxes, labels). To create an `Image` instance:

    - Image with shape **H x W x 3 (RGB)**

    - List of bounding boxes with format **[XMIN, YMIN, XMAX, YMAX]**

    - List of labels

- Before creating the `Image` instance, each image is pre-processed using the following steps:

    - Resizing images to be **512 x 512 x 3 (RGB)**

- After loading the images and annotations as a **list of `Image` objects**, obtain the following:

    - List of `tv_tensors.Image(dtype.float32)` where each tensor has shape **3 x 512 x 512**, with pixel values scaled to the range **[0,1]**

    - List of `tv_tensors.BoundingBoxes(dtype=torch.float32)` where each tensor has shape **N x 4**

    - List of `torch.tensor(dtype=torch.int64)` where each tensor has shape **N**

## Model training

1. Obtain train and test sets for images, boxes and labels using a **train-test split of 0.8**

2. Create `torch.utils.data.DataLoader(batch_size=4)` objects for both the train and test sets

    - Insufficient CUDA memory for higher `batch_size` values (GeForce RTX 3070 Ti 8GB)

3. Load the `fasterrcnn_resnet50_fpn` model with default weights and change the box_predictor head to have **3 classes**

    - Feature extractor (trained on COCO dataset) is retained

4. Shift model to GPU (if available) and obtain the parameters where `requires_grad == True`

    - Take note that model should be shifted to GPU **before obtaining parameters**, so that the parameters are on the same device

    - `requires_grad == True` indicates that PyTorch will track all operations involving the tensor and constructs a computation graph

    - A computation graph is **used during backpropagation to calculate the gradients of the loss function W.R.T. parameters that contributed to the loss**

        - $L(W, X)=\text{Loss}$ measures the difference between the model's predictions and true target values

        - $W$ are the model parameters and $X$ is the input to the model

        - Gradients are essentially the **partial derivatives of the loss function w.r.t. the parameters $W$**

5. Using the obtained parameters, create a Stochastic Gradient Descent optimizer `torch.optim.SGD(lr=1e-5, momentum=0.9, weight_decay=0.0005)`

6. For each training epoch (total 5), execute the following lines of code:

    - Compute `loss_dict`

        - A dictionary that contains values for `loss_classifier`, `loss-box_reg` and `loss_objectness`

    - Compute `loss`

        - A scalar value and that is the sum of all values in `loss_dict`

    - Compute `epoch_loss`

        - For each batch of the current epoch, add the `loss` values together

        - Before adding `loss`, need to `.cpu().detach().numpy()`. This is because numpy arrays can only operate on the CPU and **detachment is necessary to safely work with the tensor in numpy without affecting the computation graph** for back propagation later on

    - Execute `optimizer.zero_grad()`

        - Since the gradient for each parameter accumulates by default, resetting the gradients of the model parameters to 0 so that the **gradients can update correctly during back propagation**

    - Execute `loss.backward()`

        - Calculates and stores the gradients of the loss W.R.T. the model parameters

    - Execute `optimizer.step()`

        - Updates the gradients of the model parameters with the calculated gradients from `loss.backward()`

        - Optimizer will **adjust the parameters in the direction that minimizes the loss**
