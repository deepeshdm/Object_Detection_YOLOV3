import torch
import cv2
import config
import numpy as np
from model import YOLOv3
from utils import intersection_over_union, cells_to_bboxes, plot_image


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    i = 0
    while bboxes:

        if i % 100 == 0 or i != 0:
            print(i, " Bounding Boxes Processed...")

        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
               or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
               < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)
        i += 1

    return bboxes_after_nms


# Takes an Image Path & Pretrained_Weights as input and plots bounding boxes on
# objects detected on image,then displays the image back.
def detect_objects(image_path, weights_path=None):
    assert type(image_path) == str, "image_path needs to be string"
    assert type(weights_path) == str, "weights_path needs to be string"

    img = cv2.imread(image_path)
    # Input shape of Yolo V3
    input_shape = (416, 416)
    img_1 = cv2.resize(img, input_shape)

    # reshaping image as needed by pytorch
    img = np.array(img_1)
    img = img.reshape(-1, 3, input_shape[0], input_shape[1])
    img = torch.Tensor(img)

    model = YOLOv3(num_classes=20)

    # Loading pretrained weights
    if weights_path:
        print("Loading Pre-trained weights...")
        # If Gpu available
        if torch.cuda.is_available():
            checkpoint = torch.load(weights_path)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])

    # Returns a tensor of shape (N, 3, S, S, num_classes+5)
    output = model(img)

    anchors = (torch.tensor(config.ANCHORS)
               * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))

    bboxes = [[] for _ in range(output[0].shape[0])]
    for i in range(3):
        batch_size, A, S, _, _ = output[i].shape
        anchor = anchors[i]
        boxes_scale_i = cells_to_bboxes(
            output[i], anchor, S=S, is_preds=True
        )
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box

    print("Applying Non-Max Supression...")
    nms_boxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint", )

    print("Plotting BBoxes...")
    plot_image(img_1, nms_boxes)
