import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from scripts.models import Encoder
from scripts.datasets import *
from scripts.utils import *
from nltk.translate.bleu_score import corpus_bleu

# Data parameters
CURR_DIR = os.getcwd()
data_folder = CURR_DIR + '/outputs/'  # folder with data files saved by create_input_files.py
data_name = 'flickr8k_1_cap_per_img_20_min_word_freq'  # base name shared by data files


train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)