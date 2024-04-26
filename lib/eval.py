"""
	Code here heavily burrows from https://github.com/locuslab/wanda/tree/main
"""
import time
import torch
import torch.nn as nn
import pdb
from .data import get_loaders 
from .eval_gsm8k import eval_ppl_test_gsm8k, eval_ppl_train_gsm8k

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, trainenc, testenc, device=torch.device("cuda:0"), dataset="wikitext2", bsz=1):

	# Print status
	print(f"evaluating on {dataset}")

	# Get the test loader
	trainloader, testloader = get_loaders(
		dataset, trainenc, testenc, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
	)

	# Evaluate ppl in no grad context to avoid updating the model
	with torch.no_grad():
		if dataset == 'gsm8k':
			ppl_test = eval_ppl_test_gsm8k(model, testloader, bsz, device)
			ppl_train = eval_ppl_train_gsm8k(model, trainloader, bsz, device)
		else:
			ppl_test = eval_ppl_test(model, testloader, bsz, device)
			ppl_train = eval_ppl_train(model, trainloader, bsz, device)
	return ppl_train, ppl_test 

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl_trainonly(model, tokenizer, trainenc, testenc, bsz=1, nsamples=128, device=torch.device("cuda:0"), seed=0, dataset="wikitext2"):

	print(f"evaluating on {dataset}")
	# Get the test loader
	trainloader, _ = get_loaders(
		dataset, trainenc, testenc, nsamples=nsamples, seed=seed, seqlen=model.seqlen, tokenizer=tokenizer 
	)

	# Evaluate ppl in no grad context to avoid updating the model
	with torch.no_grad():
		if dataset == 'gsm8k':
			ppl_train = eval_ppl_train_gsm8k(model, trainloader, bsz, device)
		else:
			ppl_train = eval_ppl_train(model, trainloader, bsz, device)
	return ppl_train

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_train(model, trainloader, bs=1, device=None):
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
	print(f"test: nsamples {nsamples}")

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