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

\usepackage{subfigure}
\usepackage{multicol}
\usepackage{tabularx}
\usepackage{booktabs}
\usepackage{gensymb}
\usepackage{enumitem}

\begin{table}

    \caption{Comparison against the traditional approaches when evaluating on different palmprint images from IITD and CASIA datasets. The results are presented in the form of a $\mu\pm\sigma$ and in ($\%$).} 
    \centering % used for centering table

    ~\\*\begin{tabularx}{1\textwidth} { 
       >{\centering\arraybackslash}X 
       >{\centering\arraybackslash}X 
       >{\centering\arraybackslash}X
       >{\centering\arraybackslash}X 
       >{\centering\arraybackslash}X 
       >{\centering\arraybackslash}X
       >{\centering\arraybackslash}X
       >{\centering\arraybackslash}X
       >{\centering\arraybackslash}X 
       >{\centering\arraybackslash}X }
        \hline\hline 
        Method & HOG \cite{hog_original} & LBP \cite{lbp} & LPQ \cite{lpq} & RILPQ \cite{rilpq} & Gabor \cite{gabor} & BSIF \cite{bsif} & POEM \cite{poem} & Global & Two-Path\\
        \hline
        \\
        \cline{4-8}
        & \multicolumn{9}{c}{\textbf{Evaluated on the left palmprints from IITD dataset}} \\ \cline{4-8} \\

        %& \multicolumn{9}{c}{\textbf{Evaluated on the left palmprints from IITD dataset}} \\ 
        %\hline

        EER & $7.216 \pm  1.519$ & $6.380 \pm 0.577$ & $1.383 \pm 0.578$ & $1.824 \pm 0.508$ & $7.123 \pm 0.744$ & $1.121 \pm 0.431$ & $1.737 \pm 1.014$ & $1.849 \pm 0.641$ & $\boldsymbol{0.757} \pm \boldsymbol{0.212}$ \\

        AUC & $97.394 \pm 0.690$ & $98.072 \pm 0.437$ & $99.812 \pm 0.133$ & $99.660 \pm 0.289$ & $97.495 \pm 0.694$ & $99.768 \pm 0.142$ & $99.359 \pm 0.442$ & $99.770 \pm 0.127$ & $\boldsymbol{99.935} \pm \boldsymbol{0.065}$ \\

        VER@0.1FAR & $81.739 \pm 3.049$ & $71.652 \pm 1.272$ & $96.173 \pm 1.006$ &  $97.130 \pm 0.650$ & $73.739 \pm 2.730$ & $98.0 \pm 0.976$ & $96.434 \pm 1.516$ & $94.782 \pm 1.099$ & $\boldsymbol{98.086} \pm \boldsymbol{0.347}$ \\

        VER@1FAR & $88.086 \pm 2.104$ & $81.391 \pm 0.968$ & $98.521 \pm 0.706$ & $98.0 \pm 0.650$ & $85.304 \pm 1.659$ & $98.869 \pm 0.650$ & $97.913 \pm 1.358$ &  $97.478 \pm 1.179$ & $\boldsymbol{99.478} \pm \boldsymbol{0.507}$ \\
        %\hline

        \\
        \cline{4-8}
        & \multicolumn{9}{c}{\textbf{Evaluated on the right palmprints from IITD dataset}} \\ \cline{4-8} \\

        %& \multicolumn{9}{c}{\textbf{Evaluated on the right palmprints from IITD dataset}} \\ 
        %\hline

        EER & $ 6.327 \pm 1.041 $ & $ 7.209 \pm 0.488 $ & $ 1.309 \pm 0.266 $ & $ 1.101 \pm 0.428 $ & $ 6.184 \pm 0.866 $ & $ 1.191 \pm 0.453 $ & $ 1.652 \pm 0.426 $ & $ 0.900 \pm 0.421 $ & $ \boldsymbol{0.602} \pm \boldsymbol{0.234} $ \\

        AUC & $ 97.796 \pm 0.691 $ & $ 97.607 \pm 0.496 $ & $ 99.868 \pm 0.090 $ & $ 99.855 \pm 0.123 $ & $ 98.050 \pm 0.335 $ & $ 99.780 \pm 0.121 $ & $ 99.701 \pm 0.191 $ & $ 99.939 \pm 0.056 $ & $ \boldsymbol{99.912} \pm \boldsymbol{0.102} $ \\

        VER@0.1FAR & $ 82.434 \pm 0.706 $ & $ 69.304 \pm 1.307 $ & $ 96.434 \pm 0.886 $ & $ 97.565 \pm 0.706 $ & $ 75.217 \pm 1.760 $ & $ 97.565 \pm 0.706 $ & $ 96.695 \pm 0.806 $ & $ 97.391 \pm 1.626 $ & $ \boldsymbol{98.521} \pm \boldsymbol{0.589} $ \\

        VER@1FAR & $ 88.608 \pm 0.325 $ & $ 77.043 \pm 2.599 $ & $ 98.608 \pm 0.325 $ & $ 99.043 \pm 0.748 $ & $ 87.217 \pm 2.994 $ & $ 98.521 \pm 0.758 $ & $ 98.260 \pm 0.476 $ & $ 99.043 \pm 0.576 $ & $\boldsymbol{99.478} \pm \boldsymbol{0.425}$ \\

        %\hline
        \\
        \cline{4-8}
        & \multicolumn{9}{c}{\textbf{Evaluated on the left palmprints from CASIA dataset}} \\ \cline{4-8}

        %& \multicolumn{9}{c}{\textbf{Evaluated on the left palmprints from CASIA dataset}} \\ 
        %\hline

        EER & $ 5.025\pm 1.005 $ & $ 5.150 \pm 0.663 $ & $ 0.904 \pm 0.336 $ & $ 0.705 \pm 0.216 $ & $7.530 \pm 0.475 $ & $ 0.465 \pm 0.230 $ & $ 0.679 \pm 0.117 $ & $ 0.385 \pm 0.273 $ & $ \boldsymbol{0.374} \pm \boldsymbol{0.164}$ \\
        AUC & $ 98.557 \pm 0.546 $ & $ 98.992 \pm 0.174 $ & $ 99.896 \pm 0.095 $ & $ 99.947 \pm 0.060 $ & $ 96.704 \pm 0.418 $ & $ 99.969 \pm 0.039 $ & $ 99.939 \pm 0.046 $ & $ 99.987 \pm 0.011 $ & $ \boldsymbol{99.986} \pm \boldsymbol{0.015} $ \\

        VER@0.1FAR & $ 86.489 \pm 0.657 $ & $ 64.229 \pm 0.781 $ & $ 92.411 \pm 2.863 $ & $ 98.671 \pm 0.563 $ & $ 61.960 \pm 0.756 $ & $ 99.003 \pm 0.646 $ & $ 98.172 \pm 0.622 $ & $ 99.113 \pm 0.847 $ & $ \boldsymbol{99.113} \pm \boldsymbol{0.687} $ \\

        VER@1FAR & $ 91.861 \pm 1.625 $ & $ 82.889 \pm 2.163 $ & $ 98.836 \pm 0.847 $ & $ 99.501 \pm 0.322 $ & $ 80.011 \pm 2.228 $ & $ 99.723 \pm 0.303 $ & $ 99.501 \pm 0.207 $ & $ 99.722 \pm 0.303 $ & $ \boldsymbol{99.778} \pm \boldsymbol{0.207} $ \\

        %\hline
        \\
        \cline{4-8}  
        & \multicolumn{9}{c}{\textbf{Evaluated on the right palmprints from CASIA dataset}} \\ \cline{4-8} \\
        %& \multicolumn{9}{c}{\textbf{Evaluated on the right palmprints from CASIA dataset}} \\ 

        EER & $ 6.691 \pm 0.691 $ & $ 5.577 \pm 0.618 $ & $ 1.533 \pm 0.421 $ & $ 0.949 \pm 0.336 $ & $ 7.677 \pm 0.334 $ & $ 0.981 \pm 0.487 $ & $ 0.997 \pm 0.275 $ & $ 1.208 \pm 0.306 $ & $ \boldsymbol{0.553} \pm \boldsymbol{0.315} $ \\

        AUC & $ 97.756 \pm 0.234 $ & $ 98.725 \pm 0.256 $ & $ 99.659 \pm 0.030 $ & $ 99.885 \pm 0.064 $ & $ 97.065 \pm 0.471 $ & $ 99.770 \pm 0.225 $ & $ 99.695 \pm 0.167 $ & $ 99.801 \pm 0.089 $ & $ \boldsymbol{99.917} \pm \boldsymbol{0.070} $ \\

        VER@0.1FAR & $ 82.631 \pm 1.561 $ & $ 61.652 \pm 1.595 $ & $ 90.792 \pm 4.662 $ & $ 97.436 \pm 0.317 $ & $ 62.938 \pm 5.086 $ & $ 97.864 \pm 1.093 $ & $ 97.959 \pm 0.465 $ & $ 96.583 \pm 0.906 $ & $ \boldsymbol{98.625} \pm \boldsymbol{0.863} $ \\

        VER@1FAR & $ 88.041 \pm 1.516 $ & $ 80.305 \pm 1.449 $ & $ 98.101 \pm 0.721 $ & $ 98.907 \pm 0.512 $ & $ 79.879 \pm 3.645 $ & $ 99.051 \pm 0.541 $ & $ 99.003 \pm 0.274 $ & $ 98.718 \pm 0.439 $ & $ \boldsymbol{99.525} \pm \boldsymbol{0.299} $ \\

        \hline\hline

    \end{tabularx}
    \label{table:classical-methods-metrics}
\end{table}


\usepackage{booktabs}
% --
\begin{table}
	\centering
	\begin{tabular}{lcc}
		\toprule
		& \multicolumn{2}{c}{Data} \\ \cmidrule(lr){2-3}
		Name & Column 1 & Another column \\
		\midrule
		Some data & 10 & 95 \\
		Other data & 30 & 49 \\
		\addlinespace
		Different stuff & 99 & 12 \\
		\bottomrule
	\end{tabular}
	\caption{My caption.}
	\label{tab-label}
\end{table}





