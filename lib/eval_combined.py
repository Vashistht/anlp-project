"""
	Code here heavily burrows from https://github.com/locuslab/wanda/tree/main
"""
import time
import torch
import torch.nn as nn
import pdb
from .data import get_loaders 
import numpy as np
import string
import re
from collections import Counter

EXPECTED_METRIC_WEIGHTS_LENGTH = 1

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_combined(model, tokenizer, trainenc, testenc, metric_weights, device=torch.device("cuda:0"), dataset="wikitext2", bsz=1):

	# Print status
	print(f"evaluating on {dataset}")

	# Get the test loader
	trainloader, testloader = get_loaders(
		dataset, trainenc, testenc, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
	)

	ppl_weight = metric_weights[0]
	lexsim_weight = metric_weights[1]
	cossim_weight = metric_weights[2]
	acc_weight = metric_weights[3]

	# Evaluate ppl in no grad context to avoid updating the model
	with torch.no_grad():
		if dataset == 'gsm8k':
			ppl_train = eval_ppl_train_gsm8k(model, trainloader, bsz, device)
			ppl_test = eval_ppl_test_gsm8k(model, testloader, bsz, device)
			lexsim_train, cossim_train, acc_train = eval_combined_helper(model, trainloader, tokenizer, bsz, device)
			lexsim_test, cossim_test, acc_test = eval_combined_helper(model, testloader, tokenizer, bsz, device)
			# lexsim_train = eval_lexsim_gsm8k(model, trainloader, tokenizer, bsz, device)
			# lexsim_test = eval_lexsim_gsm8k(model, testenc, tokenizer, bsz, device)
			# cossim_train = eval_semantic_sim_gsm8k(model, trainloader, tokenizer, bsz, device)
			# cossim_test = eval_semantic_sim_gsm8k(model, testloader, tokenizer, bsz, device)
			# acc_train = eval_acc_gsm8k(model, trainloader, tokenizer, bsz, device)
			# acc_test = eval_acc_gsm8k(model, testloader, tokenizer, bsz, device)
		else:
			ppl_test = eval_ppl_test(model, testloader, bsz, device)
			ppl_train = eval_ppl_train(model, trainloader, bsz, device)

	combined_train = ppl_weight * ppl_train + lexsim_weight * lexsim_train + cossim_weight * cossim_train + acc_weight * acc_train
	combined_test = ppl_weight * ppl_test + lexsim_weight * lexsim_test + cossim_weight * cossim_test + acc_weight * acc_test
	return combined_train, combined_test 

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_combined_trainonly(model, tokenizer, trainenc, testenc, metric_weights, bsz=1, nsamples=128, device=torch.device("cuda:0"), seed=0, dataset="wikitext2"):

	print(f"evaluating on {dataset}")
	# Get the test loader
	trainloader, _ = get_loaders(
		dataset, trainenc, testenc, nsamples=nsamples, seed=seed, seqlen=model.seqlen, tokenizer=tokenizer 
	)

	ppl_weight = metric_weights[0]
	lexsim_weight = metric_weights[1]
	cossim_weight = metric_weights[2]
	acc_weight = weight_metrics[3]

	# Evaluate ppl in no grad context to avoid updating the model
	with torch.no_grad():
		ppl_train = eval_ppl_train_gsm8k(model, trainloader, bsz, device)
		lexsim_train = eval_lexsim_gsm8k(model, trainloader, tokenizer, device)
		cossim_train = eval_semantic_sim_gsm8k(model, trainloader, tokenizer, device)
		acc_train = eval_acc_gsm8k(model, trainloader, tokenizer, device)

	combined_train = ppl_weight * ppl_train + lexsim_weight * lexsim_train + cossim_weight * cossim_train + acc_weight * acc_train

	return combined_train

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_train(model, trainloader, bs=1, device=None):
	# Get input IDs
	# testenc = testenc.input_ids

	# Calculate number of samples
	# nsamples = testenc.numel() // model.seqlen
	nsamples = len(trainloader)

	# List to store negative log likelihoods
	nlls = []
	print(f"nsamples {nsamples}")

	# Loop through each batch
	for i in range(0,nsamples,bs):
		if i % 50 == 0:
			print(f"sample {i}")

		# Calculate end index
		j = min(i+bs, nsamples)

		# Prepare inputs and move to device
		# inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
		this_bs = min(bs, nsamples - i)
		inputs = torch.concat([trainloader[i + k][0].to(device) for k in range(this_bs)])

		inputs = inputs.reshape(j-i, model.seqlen)

		# Forward pass through the model
		lm_logits = model(inputs).logits

		# Shift logits and labels for next token prediction
		shift_logits = lm_logits[:, :-1, :].contiguous()
		shift_labels = inputs[:, 1:]

		# Compute loss
		loss_fct = nn.CrossEntropyLoss()
		loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

		# Calculate negative log likelihood
		neg_log_likelihood = loss.float() * model.seqlen * (j-i)

		# Append to list of negative log likelihoods
		nlls.append(neg_log_likelihood)

	# Compute perplexity
	ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

	# Empty CUDA cache to save memory
	torch.cuda.empty_cache()

	return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_test(model, testenc, bs=1, device=None):
	# Get input IDs
	testenc = testenc.input_ids

	# Calculate number of samples
	nsamples = testenc.numel() // model.seqlen

	# List to store negative log likelihoods
	nlls = []
	print(f"nsamples {nsamples}")

	# Loop through each batch
	for i in range(0,nsamples,bs):
		if i % 50 == 0:
			print(f"sample {i}")

		# Calculate end index
		j = min(i+bs, nsamples)

		# Prepare inputs and move to device
		inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
		inputs = inputs.reshape(j-i, model.seqlen)

		# Forward pass through the model
		lm_logits = model(inputs).logits

		# Shift logits and labels for next token prediction
		shift_logits = lm_logits[:, :-1, :].contiguous()
		shift_labels = inputs[:, 1:]

		# Compute loss
		loss_fct = nn.CrossEntropyLoss()
		loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

		# Calculate negative log likelihood
		neg_log_likelihood = loss.float() * model.seqlen * (j-i)

		# Append to list of negative log likelihoods
		nlls.append(neg_log_likelihood)

	# Compute perplexity
	ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

	# Empty CUDA cache to save memory
	torch.cuda.empty_cache()

	return ppl.item()

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

# HELPER to combine lexical + semantic + accuracy together 
# purpose: model is run once on each question
def eval_combined_helper(model, loader, tokenizer, bs=1, device=None):
	nsamples = len(loader)

	# List to store negative log likelihoods
	f1_sum = 0.0
	cos_sim = 0.0
	em_sum = 0.0
	print(f"helper: nsamples {nsamples}")

	# Loop through each batch
	for i in range(0,nsamples,bs):
		# if i % 50 == 0:
		print(f"sample {i}")
		input_ids = loader[i][0].to(device)
		# Calculate negative log likelihood
		outputs = model.generate(input_ids, max_length=(input_ids.shape[1]+100))
		outputs_decoded = tokenizer.decode(outputs[:,input_ids.size(1):][0])
		rationale = loader[i][1]
		answer = loader[i][2]
		
		f1_sum += f1(outputs_decoded, rationale, normalize_answer)
		cos_sim += cosine_sim(outputs_decoded, rationale, tokenizer)
		em_sum += em(outputs_decoded, answer, normalize_answer)

	# Empty CUDA cache to save memory
	torch.cuda.empty_cache()

	return f1_sum / nsamples, cos_sim / int(nsamples / bs), em_sum / nsamples

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

# def eval_lexsim_gsm8k(model, loader, tokenizer, bs=1, device=None):
# 	nsamples = len(loader)

# 	# List to store negative log likelihoods
# 	f1_sum = 0.0
# 	print(f"train lexical similarity: nsamples {nsamples}")

# 	# Loop through each batch
# 	for i in range(0,nsamples,bs):
# 		# if i % 50 == 0:
# 		print(f"sample {i}")
# 		input_ids = loader[i][0].to(device)
# 		# Calculate negative log likelihood
# 		outputs = model.generate(input_ids, max_length=(input_ids.shape[1]+100))
# 		outputs_decoded = tokenizer.decode(outputs[:,input_ids.size(1):][0])
# 		rationale = loader[i][1]
# 		f1_score = f1(outputs_decoded, rationale, normalize_answer)
# 		f1_sum += f1_score

# 	# Empty CUDA cache to save memory
# 	torch.cuda.empty_cache()

# 	return f1_sum / nsamples


def cosine_sim(predictions, ground_truths, tokenizer):
    # Assuming string inputs
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    embeds = tokenizer.encode(predictions, ground_truths, return_tensors='pt', padding=True, truncation=True)
    prediction_embeddings = embeds[0]
    ground_truth_embeddings = embeds[1]
    cos_sims = cos(prediction_embeddings, ground_truth_embeddings)
	# Get avg of batch
    cos_sim = cos_sims.mean()
    return cos_sim.item()


# def eval_semantic_sim_gsm8k(model, dataloader, tokenizer, bs=1, device=None):
# 	nsamples = len(dataloader)

# 	# Running avg
# 	cos_sim = 0.0

# 	for i in range(0,nsamples,bs):
# 		if i % 50 == 0:
# 			print(f"sample {i}")
# 		input_ids = dataloader[i][0].to(device)
# 		target_ids = input_ids.clone()
# 		target_ids[:, :-1] = -100 #ignore_index token
# 		# Calculate negative log likelihood
# 		outputs = model(input_ids, labels=target_ids)
# 		outputs_decoded = tokenizer.decode(outputs)
# 		rationale_decoded = tokenizer.decode(dataloader[i][1].to(device))
# 		# cosine_sim returns avg cos_sim of batch
# 		cos_sim += cosine_sim(outputs_decoded, rationale_decoded, tokenizer)

# 	# Empty CUDA cache to save memory
# 	torch.cuda.empty_cache()
# 	return cos_sim / int(nsamples / bs)

def em(prediction, ground_truth, normalize_fn):
	norm_prediction = normalize_fn(prediction)
	# this is ok bc we normalized all white spaces
	prediction_tokens = norm_prediction.split(" ")

	norm_truth = normalize_fn(ground_truth)
	if norm_truth in prediction_tokens:
		return 1.0
    
	return 0.0

# def eval_acc_gsm8k(model, loader, tokenizer, bs=1, device=None):
# 	nsamples = len(loader)

# 	# List to store negative log likelihoods
# 	em_sum = 0.0
# 	print(f"accuracy: nsamples {nsamples}")

# 	# Loop through each batch
# 	for i in range(0,nsamples,bs):
# 		# if i % 50 == 0:
# 		print(f"sample {i}")
# 		input_ids = loader[i][0].to(device)
# 		outputs = model.generate(input_ids, max_length=(input_ids.shape[1]+100))
# 		outputs_decoded = tokenizer.decode(outputs[:,input_ids.size(1):][0])
# 		answer = loader[i][2]
# 		em_score = em(outputs_decoded, answer, normalize_answer)
# 		em_sum += em_score

# 	# Empty CUDA cache to save memory
# 	torch.cuda.empty_cache()

# 	return em_sum / nsamples