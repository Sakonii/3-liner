import math
import numpy as np
import threading
import pandas as pd

from cv2 import cv2


class Inference:
    "A Wrapper for Inference"

    def __init__(
        self,
        modelDetection,
        labelFile="./labels.csv",
        file_path="./input/",
    ):
        self.img = cv2.imread(file_path)
        self.detection = modelDetection
        self.map_cols = ["idx", "labels"]
        self.labels = pd.read_csv("./labels.csv", header=None, names=self.map_cols)
        self.labels = self.labels["labels"].tolist()
        # Default window for inference
        cv2.namedWindow(winname="foto-filter")

    def batch_detection(self, toConsole=False):
        "Detect objects in a folder"  # TO DO: "Detect objects in a batch of images in a folder"
        if toConsole:
            print("predicting this may take a whiile ...")
        idxOfClasses = self.detection.predict(self.img)[:][1]
        # Get labels from indexes
        preds = list(map(self.labels.__getitem__, idxOfClasses))
        print(preds)
        # Show predictions img
        self.img_predicted = self.detection.draw_detections(self.img)
        cv2.imshow(winname="foto-filter", mat=self.img_predicted)
        # Hold screen
        cv2.waitKey()
        cv2.destroyAllWindows()

    def pre_inference(self):
        "Console Messages before interface initialization"
        print("Initializing interface ...")

    def start_inference(self):
        "Mouse-Events Ready User Interface"
        self.pre_inference()
        self.batch_detection(toConsole=True)
