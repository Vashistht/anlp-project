"""
	Code here heavily burrows from https://github.com/locuslab/wanda/tree/main
"""
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
# Load and process BoolQ dataset
def get_boolq(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset("boolq", split='train')
    testdata = load_dataset("boolq", split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['question'] + traindata['passage']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['question'] + testdata['passage']), return_tensors='pt')

    # Generate samples from training set
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

# Load and process HellaSwag dataset
def get_hellaswag(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset("hellaswag", split='train')
    testdata = load_dataset("hellaswag", split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['activity_label'] + traindata['ctx_a'] + traindata['ctx_b'] + traindata['ctx_c'] + traindata['ctx_d']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['activity_label'] + testdata['ctx_a'] + testdata['ctx_b'] + testdata['ctx_c'] + testdata['ctx_d']), return_tensors='pt')

    # Generate samples from training set
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

# Load and process WinoGrande dataset
def get_winogrande(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset("winogrande", "xl", split='train')
    testdata = load_dataset("winogrande", "xl", split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['sentence']), return_tensors='pt')

    # Generate samples from training set
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

# Load and process ARC-e dataset
def get_arc_e(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset("ai2_arc", "ARC-E", split='train')
    testdata = load_dataset("ai2_arc", "ARC-E", split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['question'] + traindata['choices']['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['question'] + testdata['choices']['text']), return_tensors='pt')

    # Generate samples from training set
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

# Load and process ARC-c dataset
def get_arc_c(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset("ai2_arc", "ARC-C", split='train')
    testdata = load_dataset("ai2_arc", "ARC-C", split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['question'] + traindata['choices']['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['question'] + testdata['choices']['text']), return_tensors='pt')

    # Generate samples from training set
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
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if 'gsm8k' in name:
        return get_gsm8k(nsamples, seed, seqlen, tokenizer)
    if 'boolq' in name:
        return get_boolq(nsamples, seed, seqlen, tokenizer)
    if 'hellaswag' in name:
        return get_hellaswag(nsamples, seed, seqlen, tokenizer)
    if 'winogrande' in name:
        return get_winogrande(nsamples, seed, seqlen, tokenizer)
    if 'arc-e' in name:
        return get_arc_e(nsamples, seed, seqlen, tokenizer)
    if 'arc-c' in name:
        return get_arc_c(nsamples, seed, seqlen, tokenizer)