import os
import glob
import torch
import torchaudio
import numpy as np
import pandas as pd
from collections import Counter
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import IPython.display as ipd

csv_info = pd.read_csv('./UrbanSound8K/metadata/UrbanSound8K.csv')
csv_info = csv_info.set_index('slice_file_name')
csv_info.head()