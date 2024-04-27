"""
	Code here heavily burrows from https://github.com/locuslab/wanda/tree/main
"""
import time
import torch
import torch.nn as nn
import pdb
from .data import get_loaders 
from collections import Counter
import re
import string

# TODO: Check why this was done the way it was
# Now: (borrowing from the lora_ft/evaluate_ppl.py file)

# Function to evaluate perplexity (ppl)
def eval_ppl_train_gsm8k(model, trainloader, bs=1, device=None):
	# Get input IDs
	# testenc = testenc.input_ids

	# Calculate number of samples
	# nsamples = testenc.numel() // model.seqlen
	nsamples = len(trainloader)

	# List to store negative log likelihoods
	nlls = []
	print(f"train: nsamples {nsamples}")

	# Loop through each batch
	for i in range(0,nsamples,bs):
		if i % 50 == 0:
			print(f"sample {i}")

		# Prepare inputs and move to device
		# inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
		# this_bs = min(bs, nsamples - i)
		# inputs = torch.concat([trainloader[i + k][0].to(device) for k in range(this_bs)])
		# inputs = inputs.reshape(j-i, model.seqlen)

		# # Forward pass through the model
		# lm_logits = model(inputs).logits

		# # Shift logits and labels for next token prediction
		# shift_logits = lm_logits[:, :-1, :].contiguous()
		# shift_labels = inputs[:, 1:]

		# # Compute loss
		# loss_fct = nn.CrossEntropyLoss()
		# # import pdb; pdb.set_trace()
		# loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

		# # Calculate negative log likelihood
		# neg_log_likelihood = loss.float() * model.seqlen * (j-i)

		# # Append to list of negative log likelihoods
		# nlls.append(neg_log_likelihood)
		input_ids = trainloader[i][0].to(device)
		target_ids = input_ids.clone()
		target_ids[:, :-1] = -100 #ignore_index token
		# Calculate negative log likelihood
		outputs = model(input_ids, labels=target_ids)
		loss = outputs.loss
		neg_log_likelihood = loss.float()
		nlls.append(neg_log_likelihood)
	# Compute perplexity
	# ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

	ppl = torch.exp(torch.stack(nlls).mean())
	# Empty CUDA cache to save memory
	torch.cuda.empty_cache()

	return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_test_gsm8k(model, testenc, bs=1, device=None):
	# Get input IDs
	# testenc = testenc.input_ids
	# Calculate number of samples
	# nsamples = testenc.numel() // model.seqlen #@vashistht: this is not true for gsm8k sort of dataset which depends on each question
	nsamples = len(testenc)
	# List to store negative log likelihoods
	nlls = []
	print(f"test: nsamples {nsamples}")

	# Loop through each batch
	for i in range(0,nsamples,bs):
		if i % 50 == 0:
			print(f"sample {i}")

		# Calculate end index
		# j = min(i+bs, nsamples)

		# Prepare inputs and move to device
		# inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)

		# inputs = inputs.reshape(j-i, model.seqlen)

		# # Forward pass through the model
		# lm_logits = model(inputs).logits

		# # Shift logits and labels for next token prediction
		# shift_logits = lm_logits[:, :-1, :].contiguous()
		# shift_labels = inputs[:, 1:]

		# # Compute loss
		# loss_fct = nn.CrossEntropyLoss()
		# loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
		input_ids = testenc[i][0].to(device)
		target_ids = input_ids.clone()
		target_ids[:, :-1] = -100 #ignore_index token
		# Calculate negative log likelihood
		outputs = model(input_ids, labels=target_ids)
		loss = outputs.loss
		neg_log_likelihood = loss.float()

		# Append to list of negative log likelihoods
		nlls.append(neg_log_likelihood)

	# Compute perplexity
	# ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
	ppl = torch.exp(torch.stack(nlls).mean() )

	# Empty CUDA cache to save memory
	torch.cuda.empty_cache()

	return ppl.item()

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1(prediction, ground_truth, normalize_fn):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def eval_lexsim_train_gsm8k(model, trainloader, tokenizer, bs=1, device=None):
	nsamples = len(trainloader)

	# List to store negative log likelihoods
	f1_sum = 0.0
	print(f"train lexical similarity: nsamples {nsamples}")

	# Loop through each batch
	for i in range(0,nsamples,bs):
		if i % 50 == 0:
			print(f"sample {i}")

		input_ids = trainloader[i][0].to(device)
		target_ids = input_ids.clone()
		target_ids[:, :-1] = -100 #ignore_index token
		# Calculate negative log likelihood
		outputs = model(input_ids, labels=target_ids)
		outputs_decoded = tokenizer.decode(outputs)
		rationale_decoded = tokenizer.decode(trainloader[i][1].to(device))
		f1 = f1(outputs_decoded, rationale_decoded, normalize_answer)
		f1_sum += f1

	# Empty CUDA cache to save memory
	torch.cuda.empty_cache()

	return f1_sum / nsamples

def eval_lexsim_test_gsm8k(model, testenc, tokenizer, bs=1, device=None):
	nsamples = len(testenc)

	# List to store negative log likelihoods
	f1_sum = 0.0
	print(f"test lexical similarity: nsamples {nsamples}")

	# Loop through each batch
	for i in range(0,nsamples,bs):
		if i % 50 == 0:
			print(f"sample {i}")

		input_ids = testenc[i][0].to(device)
		target_ids = input_ids.clone()
		target_ids[:, :-1] = -100 #ignore_index token
		# Calculate negative log likelihood
		outputs = model(input_ids, labels=target_ids)
		outputs_decoded = tokenizer.decode(outputs)
		rationale_decoded = tokenizer.decode(testenc[i][1].to(device))
		f1 = f1(outputs_decoded, rationale_decoded, normalize_answer)
		f1_sum += f1

	# Empty CUDA cache to save memory
	torch.cuda.empty_cache()

	return f1_sum / nsamples