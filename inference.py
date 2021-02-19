import glob
import numpy as np
import pandas as pd

from cv2 import cv2
from tqdm import tqdm


class Inference:
    "A Wrapper for Inference"

    def __init__(
        self,
        modelDetection,
        labelFile="./labels.csv",
        folderPath="./input/",
    ):
        # List of image-paths from folderPath
        self.imgPathList = pd.Series(glob.glob(f"{folderPath}*.jpg", recursive=True))
        # print(self.imageList)
        self.detection = modelDetection
        self.labels = pd.read_csv("./labels.csv", header=None, names=["idx", "labels"])
        self.labels = self.labels["labels"].tolist()
        # Default window for inference

    def show_img_with_preds(self, img, preds):
        "Shows predicted labels in image"
        img_predicted = self.detection.draw_detections(img)
        cv2.namedWindow(winname="foto-filter")
        cv2.imshow(winname="foto-filter", mat=img_predicted)
        cv2.waitKey(500)

    def destroy_cv2_windows(self):
        "Hold the screen for key input and destroy the windows"
        print("Done. Press any key to continue ...")
        cv2.waitKey()
        cv2.destroyAllWindows()

    def detect(self, imgPath):
        "Detect objects in an image"
        self.img = cv2.imread(imgPath)
        idxOfPredictedClasses = self.detection.predict(self.img)[:][1]
        # Get labels from indexes
        preds = list(map(self.labels.__getitem__, idxOfPredictedClasses))
        print(preds)
        self.show_img_with_preds(img=self.img, preds=preds)
        return preds

    def batch_detection(self):
        "Detect objects in a batch of images in a folder"
        for imgPath in tqdm(self.imgPathList):
            self.detect(imgPath)
        

    def pre_inference(self):
        "Console Messages before interface initialization"
        print("Initializing interface ...")

    def start_inference(self):
        "Mouse-Events Ready User Interface"
        self.pre_inference()
        self.batch_detection()
