from cv2 import cv2

import numpy as np
import re


def text_filter(txt):
    "Filter text to match model"
    txt = txt.lower()
    return re.sub("[^A-Za-z0-9 ]+", "", txt)


def draw_contours(contours, img, color=(75, 150, 0)):
    "Draws list of contours"
    cv2.drawContours(
        image=img,
        contours=contours,
        contourIdx=-1,
        color=color,
        thickness=2,
        lineType=cv2.LINE_AA,
    )


def remove_np_array_from_list(listOfArray, npArray):
    # Custom List Item Removal Function
    ind = 0
    size = len(listOfArray)
    while ind != size and not np.array_equal(listOfArray[ind], npArray):
        ind += 1
    if ind != size:
        listOfArray.pop(ind)
    else:
        raise ValueError("array not found in list.")
