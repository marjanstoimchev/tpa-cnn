import copy
import pandas as pd
import numpy as np
import sys
import os
import itertools
import glob
import math
import cv2
import time
import random
import argparse
from itertools import chain, groupby
from collections import defaultdict
import gc, warnings, joblib
from tqdm import tqdm
from PIL import Image, ImageOps
import tqdm.notebook as tq
from scipy.special import softmax
from imgaug import augmenters as iaa

from  builtins import any as b_any
from collections import OrderedDict
from prettytable import PrettyTable

import torch
from torch.cuda import amp
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import ConcatDataset
from torch.cuda.amp import autocast, GradScaler


import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as transforms


from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
