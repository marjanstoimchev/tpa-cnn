# tpa-cnn

This repository contains PyTorch implementation of the following paper: 
Stoimchev, M., Ivanovskа, M., Štruc, V. Learning to Combine Local and Global Image Information for Contactless Palmprint Recognition
(This is an initial version, the code will be updated)

# The Two-Path Architecture (TPA)

<img id="photo1" style="height:512px;width:auto;" src="media/tpa_model.png" height="512" />

##  Table of Contents
- [The Two-Path Architecture (TPA)](#TPA)
    - [Installation](#installation)
    - [Training](#training)
    - [Testing](#testing)
    - [Citing Two-Path Architecture](#citing-tpa)
    - [Reference](#reference)

## Dependencies

- Python3
- PyTorch
- Torchvision
- Numpy
- Matplotlib
- Pillow / PIL
- imgaug 
- scikit-learn
## Installation
1. First clone the repository
   ```
   git clone https://github.com/Marjan1111/tpa-cnn.git
   ```
2. Create the virtual environment via conda
    ```
    conda create -n tpa python=3.9
    ```
3. Activate the virtual environment.
    ```
    conda activate tpa
    ```
3. Install the dependencies.
   ```
   pip install -r requirements.txt
   ```
## Training
To list the arguments, run the following command:
```
python main.py -h
```

### Example how to train on IITD dataset

```
python main.py \     
    --backbone Vgg \         
    --data IITD \  # switch to: CASIA if you want to train on CASIA dataset      
    --palm_train left \  # note: if you chose left forr train, chose right for palm_test, and vice versa.
    --palm_test right \ 
    --n_epochs 100 \  
    --num_trainable 10 \ 
    --metric_head arc_margin \ 
    --patches [75, 1, 0, 30] \ 
    --lr_centers 0.5 \ 
    --alpha 0.001 \ 
    --save_path saved_models \ 
    --model_type Vgg_16 \ 
```

## Testing
To start the all-vs-all evaluation protocol, run the following command:

```bat
python test.py
```


### How to create the file structure for IITD and CASIA datasets

```
datasets
├── IITD_ROI
│   ├── Segmented
│          ├── Left
|          |   └── 001_1.bmp
|          |   ...
|          |   └── 230_5.bmp
|          |
│          ├── Right
|              └── 001_1.bmp
|              ...
|              └── 230_5.bmp
|
├── CASIA_ROI
        └── 0001_m_l_01.jpg
        ...
        └── 0312_m_r_11.jpg
```
