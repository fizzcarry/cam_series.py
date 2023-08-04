import numpy as np
import os
import scipy.stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from numpy import unique
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.manifold import TSNE
import sklearn
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
import time
import math
from scipy import signal
import warnings
import random
g=9.68
spilt_num=1024
step_num=256
