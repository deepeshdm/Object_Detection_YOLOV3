'''
==============
IMPORTANT NOTE :
=============

This is an Implementation of Object Detection using YOLO V3 trained on Pascal VOC Dataset.
==> MODEL SUMMARY :
Total params: 61,626,049
Trainable params: 61,626,049
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.98
Forward/backward pass size (MB): 1228.70
Params size (MB): 235.08
Estimated Total Size (MB): 1465.77
----------------------------------------------------------------

detection_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                     'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                     'dog', 'horse', 'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

This model can detect only the objects listed in the above detection list.

For each Image YOLO V3 predicts 10,647 bboxes and then applies non-max supression to choose the best bboxes.

==> THE MODEL MAY SOMETIMES PRODUCE INCORRECT PREDICTIONS & NEEDS TO TRAINED
FOR MORE EPOCHS TO INCREASE ACCURACY.

==> Images are resized to 416x416 since the model accepts a fixed size image as input.

Each bounding box displays the object name it has detected and the
probability of an object being present inside the bounding box.

(GPU RECOMMENDED FOR RUNNING THE MODEL)

'''

# -------------------------------------------------------------------------

'''
Below is an simple API function of the model which takes the location of your image as
input & then displays the image back with detected objects represented under bounding boxes.

The function also takes an important parameter 'weights_path' which is the path where the 
pretrained weights of the model are located.Given a path it'll load 
those learned parameters into the model (IMPORTANT TO SPECIFY ELSE MODEL WONT WORK)
'''

# Insert the location of your Image below
IMAGE_PATH = r'C:\Users\dipesh\Desktop\Pascal Voc\images\000035.jpg'

# Insert the location of weights file below
WEIGHTS_PATH = r'C:\Users\dipesh\Desktop\yolov3_pascal_78.1map.pth.tar'

from Yolo_API import detect_objects

detect_objects(image_path=IMAGE_PATH, weights_path=WEIGHTS_PATH)
