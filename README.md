# foto-filter

Search Image in a folder by image-object

## Requirements and dependencies

Dependencies installation under pip package manager

``` bash
python3 -m pip install -r requirements.txt
# If requirements.txt fails: 
# Pytorch: https://pytorch.org/get-started/locally/
# Detectron2: https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md
```

## Inference

``` bash
python3 main.py
```

# CLI Arguments:
* '--folderPath' : Folder to search for images Default = './input'
* '--modelDetection' : Filename of weights associated with detection
* '--cfgPath' : Path to detectron model cfg file relative to 'detectron2/model_zoo'
* '--labelFile' : Path to model labels (csv file)
```
