import sys
import os
from time import sleep
from openai import OpenAI
import json
import pickle
from config import *
from transformers import BertTokenizer, BertModel
import torch
import re
from sentence_transformers import SentenceTransformer

def check_data_abs(name="data"):
    return os.path.exists(f"{name}.pkl")

def dump_data_abs(data, name="data"):
    with open(f"{name}.pkl", "wb") as file:
        pickle.dump(data, file)

def load_data_abs(name="data"):
    if not os.path.exists(f"{name}.pkl"):
        return None
    with open(f"{name}.pkl", "rb") as file:
        data = pickle.load(file)
    return data