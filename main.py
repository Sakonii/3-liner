import argparse

from detection import Detection
from inference import Inference


def main():
    detection = Detection(modelWeights=args.model_detection, cfgPath=args.cfg_path)
    Inference(
        file_path=args.folder_path,
        labelFile=args.labelFile,
        modelDetection=detection,
    ).start_inference()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image File Name")
    parser.add_argument(
        "--folder_path",
        type=str,
        default="./input/img.jpg",
        help="Enter the Input camera number. Default=0; May vary depending on number of cameras connected.",
    )
    parser.add_argument(
        "--model_detection",
        type=str,
        default="http://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl",
        help="Pre-trained Weights for Detectron Detection",
    )
    parser.add_argument(
        "--cfg_path",
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
