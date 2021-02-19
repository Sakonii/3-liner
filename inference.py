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
        self.detection = modelDetection
        # Load detection labels
        self.labels = pd.read_csv("./labels.csv", header=None, names=["idx", "labels"])
        self.labels = self.labels["labels"].tolist()
        # Output dataframe declaration (empty)
        self.df = pd.DataFrame(columns=["imgPath", "preds"])
        # List of image's-path from folderPath
        self.imgPathList = pd.Series()
        for fileType in ["*.png", "*.JPEG", "*.jpg"]:
            self.imgPathList = self.imgPathList.append(
                pd.Series(glob.glob(f"{self.folderPath}/{fileType}", recursive=True))
            )

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

    def post_inference(self, debug=True):
        "Hold the screen for key input and destroy the windows"
        print("Done. Press any key to continue ...")
        # Save preds to file
        self.df.to_feather(
            f"./preds/{self.folderPath.replace('/', '').replace('.', '')}.feather"
        )
        if debug:
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

    def search(self):
        search_strs = input(">> ").split()
        final_list = []
        final_resized_images = []
        width = int(500)
        height = int(400)
        dim = (width, height)
        for search_str in search_strs:
            img_list = self.df['preds']
            for i, item in enumerate(img_list):
                print(item)
                if search_str in item:
                    print(self.df['imgPath'][i])
                    if self.df['imgPath'][i] not in final_list:
                        final_list.append(self.df['imgPath'][i])
        for ite in final_list:
            cv2.namedWindow(winname="helo")
            img = cv2.imread(ite, cv2.IMREAD_UNCHANGED)
            img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            final_resized_images.append(img_resized)

        hori = np.concatenate(final_resized_images, axis=1)
        cv2.imshow(winname="helo", mat=hori)
        cv2.waitKey()
        cv2.destroyAllWindows()


    def start_inference(self):
        "Mouse-Events Ready User Interface"
        self.pre_inference()
        self.batch_detection()
        self.post_inference()
        self.search()
