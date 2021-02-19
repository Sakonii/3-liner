import glob
import numpy as np
import pandas as pd
import os.path

from cv2 import cv2
from tqdm import tqdm


class Inference:
    "A Wrapper for Inference"

    def __init__(
        self,
        modelDetection,
        labelFile="./labels.csv",
        folderPath="./input",
    ):
        self.folderPath = folderPath
        # List of image's-path from folderPath
        self.imgPathList = pd.Series(
            glob.glob(f"{self.folderPath}/*.jpg", recursive=True)
        )
        self.detection = modelDetection
        # Load detection labels
        self.labels = pd.read_csv("./labels.csv", header=None, names=["idx", "labels"])
        self.labels = self.labels["labels"].tolist()
        # Output dataframe declaration (empty)
        self.df = pd.DataFrame(columns=["imgPath", "preds"])

    def show_img_with_preds(self, img, preds):
        "Shows predicted labels in image"
        img_predicted = self.detection.draw_detections(img)
        cv2.namedWindow(winname="foto-filter")
        cv2.imshow(winname="foto-filter", mat=img_predicted)
        cv2.waitKey(500)

    def detect(self, imgPath, debug=True):
        "Detect objects in an image"
        self.img = cv2.imread(imgPath)
        idxOfPredictedClasses = self.detection.predict(self.img)[:][1]
        # Get labels from indexes
        preds = list(set(map(self.labels.__getitem__, idxOfPredictedClasses)))
        if debug:
            print(preds)
            self.show_img_with_preds(img=self.img, preds=preds)
        return preds

    def batch_detection(self):
        "Detect objects in a batch of images in a folder"
        for imgPath in tqdm(self.imgPathList):
            # Check if image already predicted
            if not imgPath in self.df.imgPath.to_list():
                preds = self.detect(imgPath)
                self.df = self.df.append(
                    {"imgPath": imgPath, "preds": preds}, ignore_index=True
                )

    def post_inference(self):
        "Hold the screen for key input and destroy the windows"
        print("Done. Press any key to continue ...")
        # Save preds to file
        self.df.to_feather(
            f"./preds/{self.folderPath.replace('/', '').replace('.', '')}.feather"
        )
        cv2.waitKey()
        cv2.destroyAllWindows()

    def pre_inference(self):
        "Console message before interface initialization"
        # Check if folder already predicted:
        featherFilePath = (
            f"./preds/{self.folderPath.replace('/', '').replace('.', '')}.feather"
        )
        if os.path.isfile(featherFilePath):
            # predict only un-predicted files
            self.df = pd.read_feather(featherFilePath)

        print("Initializing interface ...")

    def start_inference(self):
        "Mouse-Events Ready User Interface"
        self.pre_inference()
        self.batch_detection()
        self.post_inference()
