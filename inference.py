import glob
import numpy as np
import pandas as pd
import os.path

from cv2 import cv2
from tqdm import tqdm
import eel


class UI:
    def __init__(self):
        self.output_url = []
        eel.init('../foto-filter')

    @eel.expose
    def test(msg):
        print(msg)
        return ui.output_url
ui = UI()


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

    def show_imgs(self, listOfImgPaths):
        "Shows images from list of image paths"
        listOfImgs = []
        for imgPath in listOfImgPaths:
            cv2.namedWindow(winname="foto-filter")
            self.img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
            self.img = cv2.resize(self.img, (500, 400), interpolation=cv2.INTER_AREA)
            listOfImgs.append(self.img)

        # Concatenate images horizontally:
        self.img = np.concatenate(listOfImgs, axis=1)
        cv2.imshow(winname="foto-filter", mat=self.img)
        cv2.waitKey(500)
        # Hold the screen
        cv2.waitKey()
        cv2.destroyAllWindows()

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

    def post_detection(self, debug=True):
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



    def search(self, debug=True):
        "Search inference for text input"
        searchText = input("> ")
        # For search words, add score for each matching label
        self.df["score"] = 0
        for word in searchText.split():
            self.df["score"] += self.df.preds.map(
                lambda preds: 1 if word in preds else 0
            )
        # Filter top 5 results
        self.df.sort_values(by=["score"], ascending=False, inplace=True)
        queriedResults = self.df[self.df["score"] != 0]
        outputs = queriedResults.imgPath.head(5)

        if debug:
            print(queriedResults)
            self.show_imgs(outputs)
        for it in outputs.to_list():
            ui.output_url.append(it)
        eel.start('main.html', size=(1200,700), host='localhost', port=5000)

    def start_inference(self):
        "Mouse-Events Ready User Interface"
        self.pre_inference()
        self.batch_detection()
        self.post_detection(debug=True)
        self.search(debug=True)

