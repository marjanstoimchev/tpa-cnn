# tpa-cnn
Learning to Combine Local and Global Image Information for Contactless Palmprint Recognition

## This is a model and training details repository for the following article:

Stoimchev, M., Ivanovskа, M., Štruc, V. Learning to Combine Local and Global Image Information for Contactless Palmprint Recognition

# The Two-Path Architecture (TPA)

<img id="photo1" style="height:512px;width:auto;" src="media/tpa-cnn.png" height="512" />

# Abstract

<div align="center" width="">
Among the existing biometric technologies, the field of palmprint recognition has attracted a considerable amount of attention recently because of its effectiveness, usability, and acceptability. In the past few years there has been a leap from traditional palmprint recognition methodologies, which use handcrafted features, to deep learning approaches that are able to automatically learn feature representations from the input data. However, the information that is extracted from such deep learning models typically corresponds to global image appearance, where only the most discriminative cues from the input image are considered. This characteristic is especially problematic when data is acquired in unconstrained setting, as in the case of contactless palmprint recognition systems, where visual artifacts caused by elastic deformations of the palmar surface are typically present in parts of the captured images. In this study we address the problem of elastic deformations by introducing a new approach to contactless palmprint recognition based on a specially devised CNN model. The model is designed as a two-path architecture, where one path processes the input in a holistic manner, while the second path extracts local information from smaller image patches sampled from the input image. Because elastic deformations can be assumed to most significantly affect global appearance, while having a lesser impact on spatially local image areas, the local processing path addresses the issues related to elastic deformations thereby supplementing the information from the global processing path. At the final stage of the local processing path, the most relevant information is extracted through a novel pooling operation, called channel-wise attention pooling, which is also integrated into the proposed model. The attention pooling principle is used to guide the model to focus on the most discriminative (local) parts of the input, while simultaneously giving less importance to less discriminative parts. The model is trained with a  learning objective that combines the Additive Angular Margin Loss (ArcFace) loss and the wel--known center loss. By using the proposed model design, the discriminative power of the learned image representation is significantly enhanced compared to standard holistic models, which, as we show in the experimental section, leads to state-of-the-art performance for contactless palmprint recognition. Our approach is tested on two publicly available contactless palmprint databsets, namely, IITD and CASIA, and is shown to perform favorably against state-of-the-art methods from the literature.
</div>

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

```bat

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









