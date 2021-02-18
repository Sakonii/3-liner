from cv2 import cv2

import numpy as np


def draw_contours(self, contours, img, color=(75, 150, 0)):
    "Draws list of contours"
    cv2.drawContours(
        image=img,
        contours=contours,
        contourIdx=-1,
        color=color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def draw_bboxes(self, img, color=(0, 200, 255)):
    "Draws rectangle from top_left and bottom_right co-ordinates at self.detection.bboxes"
    for bbox in self.detection.bboxes:
        top_left = bbox[0]
        bottom_right = bbox[1]
        # Default Color: Yellow
        cv2.rectangle(
            img,
            top_left,
            bottom_right,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )


def remove_np_array_from_list(self, listOfArray, npArray):
    # Custom List Item Removal Function
    ind = 0
    size = len(listOfArray)
    while ind != size and not np.array_equal(listOfArray[ind], npArray):
        ind += 1
    if ind != size:
        listOfArray.pop(ind)
    else:
        raise ValueError("array not found in list.")