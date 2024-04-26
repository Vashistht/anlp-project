"""
	Code here heavily burrows from https://github.com/locuslab/wanda/tree/main
"""
import time
import torch
import torch.nn as nn
import pdb
from .data import get_loaders 


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