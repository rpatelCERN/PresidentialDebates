import os
os.system("pip install -q tensorflow==2.3.0")
os.system("git clone --depth 1 -b v2.3.0 https://github.com/tensorflow/models.git")
os.system("pip install -Uqr models/official/requirements.txt")
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sys
sys.path.append('models')
from official.nlp.data import classifier_data_lib
from official.nlp.bert import tokenization
from official.nlp import optimization
