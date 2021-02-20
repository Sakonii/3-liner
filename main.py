import argparse

from detection import Detection
from inference import Inference


def main():
    detection = Detection(modelWeights=args.modelDetection, cfgPath=args.cfgPath)
    Inference(
        folderPath=args.folderPath,
        labelFile=args.labelFile,
        modelDetection=detection,
    ).start_inference(debug=True)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image File Folder")
    parser.add_argument(
        "--folderPath",
        type=str,
        default="./input",
        help="Folder to search for images",
    )
    parser.add_argument(
        "--modelDetection",
        type=str,
        default="http://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl",
        help="Pre-trained Weights for Detectron Detection",
    )
    parser.add_argument(
        "--cfgPath",
        type=str,
        default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        help="Path to model cfg file relative to 'detectron2/model_zoo/configs' ",
    )
    parser.add_argument(
        "--labelFile",
        type=str,
        default="./labels.csv",
        help="Path to model labels",
    )

    args = parser.parse_args()

    main()
