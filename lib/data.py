

# """ Code here heavily borrows from https://github.com/locuslab/wanda/tree/main """
# # Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py
# import numpy as np
# import random
# import torch
# import pdb
# from datasets import load_dataset
# from datasets.utils import disable_progress_bar
# disable_progress_bar()

# # Set seed for reproducibility
# def set_seed(seed):
#     np.random.seed(seed)
#     torch.random.manual_seed(seed)

# # Wrapper for tokenized input IDs
# class TokenizerWrapper:
#     def __init__(self, input_ids):
#         self.input_ids = input_ids

# # Load datasets
# def load_wikitext2():
#     traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
#     testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
#     return traindata, testdata

# def load_c4():
#     traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
#     valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
#     return traindata, valdata

# def load_gsm8k():
#     traindata = load_dataset("gsm8k", "main", split='train')
#     testdata = load_dataset("gsm8k", "main", split='test')
#     return traindata, testdata

# def get_raw_dataset(name):
#     if 'wikitext2' in name:
#         return load_wikitext2()
#     if "c4" in name:
#         return load_c4()
#     if 'gsm8k' in name:
#         return load_gsm8k()

# # Generate random subsets from datasets
# def get_wikitext2(traindata, testdata, nsamples, seed, seqlen, tokenizer):
#     random.seed(seed)
#     trainloader = []
#     for _ in range(nsamples):
#         while True:
#             i = random.randint(0, len(traindata) - 1)
#             trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
#             if trainenc.input_ids.shape[1] > seqlen:
#                 break
#         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
#         j = i + seqlen
#         inp = trainenc.input_ids[:, i:j]
#         tar = inp.clone()
#         tar[:, :-1] = -100
#         trainloader.append((inp, tar))
    
#     testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
#     return trainloader, testenc

# def get_c4(traindata, valdata, nsamples, seed, seqlen, tokenizer):
#     random.seed(seed)
#     trainloader = []
#     for _ in range(nsamples):
#         while True:
#             i = random.randint(0, len(traindata) - 1)
#             trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
#             if trainenc.input_ids.shape[1] > seqlen:
#                 break
#         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
#         j = i + seqlen
#         inp = trainenc.input_ids[:, i:j]
#         tar = inp.clone()
#         tar[:, :-1] = -100
#         trainloader.append((inp, tar))

#     valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
#     valenc = valenc.input_ids[:, :(256 * seqlen)]
#     valenc = TokenizerWrapper(valenc)
#     return trainloader, valenc

# # def get_gsm8k(traindata, testdata, nsamples, seed, seqlen, tokenizer):
# #     random.seed(seed)
# #     trainloader = []
# #     for _ in range(nsamples):
# #         while True:
# #             i = random.randint(0, len(traindata) - 1)
# #             trainenc = tokenizer(traindata[i]['question'], return_tensors='pt')
# #             if trainenc.input_ids.shape[1] > seqlen:
# #                 break
# #         i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
# #         j = i + seqlen
# #         inp = trainenc.input_ids[:, i:j]
# #         tar = inp.clone()
# #         tar[:, :-1] = -100
# #         trainloader.append((inp, tar))
    
# #     testenc = tokenizer("\n\n".join(testdata['question']), return_tensors='pt')
# #     return trainloader, testenc
# def get_gsm8k(traindata, testdata, nsamples, seed, seqlen, tokenizer):
#     random.seed(seed)
#     trainloader = []
#     for _ in range(nsamples):
#         while True:
#             i = random.randint(0, len(traindata) - 1)
#             sample = traindata[i]  # Get the complete sample

#             # Encode question, rationale, and answer separately
#             question_enc = tokenizer(sample['question'], return_tensors='pt')
#             rationale_enc = tokenizer(sample['rationale'], return_tensors='pt')
#             answer_enc = tokenizer(str(sample['answer']), return_tensors='pt') 

#             # Check if any encoding exceeds seqlen
#             if any(enc.input_ids.shape[1] > seqlen for enc in [question_enc, rationale_enc, answer_enc]):
#                 continue  # Try another sample if any exceeds seqlen

#             # Truncate if necessary (adjust truncation strategy as needed)
#             for enc in [question_enc, rationale_enc, answer_enc]:
#                 enc.input_ids = enc.input_ids[:, :seqlen]

#             # Create target tensors with -100 for masked positions
#             question_tar = question_enc.input_ids.clone()
#             question_tar[:, :-1] = -100
#             rationale_tar = rationale_enc.input_ids.clone()
#             rationale_tar[:, :-1] = -100
#             answer_tar = answer_enc.input_ids.clone()
#             answer_tar[:, :-1] = -100

#             # Append a tuple with question, rationale, and answer encodings and targets
#             trainloader.append((question_enc, rationale_enc, answer_enc, question_tar, rationale_tar, answer_tar))
#             break  # Exit the loop if a valid sample is found

#     # Process test data similarly (adjust as needed for your use case)
#     test_questions = [testdata[i]['question'] for i in range(len(testdata))]
#     test_rationales = [testdata[i]['rationale'] for i in range(len(testdata))]
#     test_answers = [str(testdata[i]['answer']) for i in range(len(testdata))]

#     test_question_enc = tokenizer(test_questions, return_tensors='pt', padding=True)
#     test_rationale_enc = tokenizer(test_rationales, return_tensors='pt', padding=True)
#     test_answer_enc = tokenizer(test_answers, return_tensors='pt', padding=True)
#     testloader = (test_question_enc, test_rationale_enc, test_answer_enc)
#     return trainloader, testloader

# # (question_enc, rationale_enc, answer_enc, question_tar, rationale_tar, answer_tar) = trainloader


# # Function to select the appropriate loader based on dataset name
# def get_loaders(name, traindata, testdata, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
#     if 'wikitext2' in name:
#         return get_wikitext2(traindata, testdata, nsamples, seed, seqlen, tokenizer)
#     if "c4" in name:
#         return get_c4(traindata, testdata, nsamples, seed, seqlen, tokenizer)
#     if 'gsm8k' in name:
#         return get_gsm8k(traindata, testdata, nsamples, seed, seqlen, tokenizer)


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
        # trainenc, testenc = load_wikitext2(tokenizer)
        return get_wikitext2(trainenc, testenc, nsamples, seed, seqlen)
    if "c4" in name:
        # traindata, valdata = load_c4(tokenizer)
        return get_c4(trainenc, testenc, nsamples, seed, seqlen, tokenizer)
    if 'gsm8k' in name:
        # trainenc, testenc = load_gsm8k(tokenizer)
        return get_gsm8k(trainenc, testenc, nsamples, seed, seqlen)