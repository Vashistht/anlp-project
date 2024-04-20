""" Code here heavily borrows from https://github.com/locuslab/wanda/tree/main """
# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py
import numpy as np
import random
import torch
import pdb
from datasets import load_dataset
from datasets.utils import disable_progress_bar
disable_progress_bar()

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load datasets
def load_wikitext2(tokenizer):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    return trainenc, testenc

def load_c4(tokenizer):
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    return traindata, valdata

def load_gsm8k(tokenizer):
    traindata = load_dataset("gsm8k", "main", split='train')
    testdata = load_dataset("gsm8k", "main", split='test')
    trainenc = tokenizer(" ".join(traindata['question']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['question']), return_tensors='pt')
    return trainenc, testenc

def get_raw_dataset(name, tokenizer):
    if 'wikitext2' in name:
        # trainenc, testenc = load_wikitext2(tokenizer)
        return load_wikitext2(tokenizer)
    if "c4" in name:
        # traindata, valdata = load_c4(tokenizer)
        return load_c4(tokenizer)
    if 'gsm8k' in name:
        # trainenc, testenc = load_gsm8k(tokenizer)
        return load_gsm8k(tokenizer)

# Generate random subsets from datasets
def get_wikitext2(trainenc, testenc, nsamples, seed, seqlen):
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(traindata, valdata, nsamples, seed, seqlen, tokenizer):
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def get_gsm8k(trainenc, testenc, nsamples, seed, seqlen):
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, trainenc, testenc, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(trainenc, testenc, nsamples, seed, seqlen)
    if "c4" in name:
        return get_c4(trainenc, testenc, nsamples, seed, seqlen, tokenizer)
    if 'gsm8k' in name:
        return get_gsm8k(trainenc, testenc, nsamples, seed, seqlen)
    
def get_traintestloader(name, tokenizer, nsamples=128, seed=0, seqlen=2048):
    trainenc, testenc = get_raw_dataset(name, tokenizer)
    return get_loaders(name, trainenc, testenc, nsamples, seed, seqlen, tokenizer)