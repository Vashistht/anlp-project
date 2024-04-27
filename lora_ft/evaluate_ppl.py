from transformers import AutoTokenizer 
from transformers import AutoModelForCausalLM
from datasets import load_dataset
import torch
import torch.nn as nn 
from peft import PeftModel, PeftConfig 
from tqdm import tqdm
import sys 
import json
import time  
import os 
from time import time
import fnmatch
import re

"""
	Code here heavily burrows from https://github.com/locuslab/wanda/tree/main
"""

# taken from lib/eval_gsm8k.py code
def eval_ppl_test_gsm8k(model, testenc, bs=1, device=None):
	nsamples = len(testenc)
	nlls = []
	total_time, total_iters = 0, 0
	for i in range(0,nsamples,bs):
		if i % 50 == 0:
			print(f"sample {i}")
		
		input_ids = testenc[i][0].to(device)
		target_ids = input_ids.clone()
		target_ids[:, :-1] = -100 #ignore_index token
		# Calculate negative log likelihood
		start_ = time()
		outputs = model(input_ids, labels=target_ids)
		total_time += (time() - start_)
		total_iters += 1
		loss = outputs.loss
		neg_log_likelihood = loss.float()

		# Append to list of negative log likelihoods
		nlls.append(neg_log_likelihood)

	# Compute perplexity
	# ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
	ppl = torch.exp(torch.stack(nlls).mean() )

	# Empty CUDA cache to save memory
	torch.cuda.empty_cache()

	return ppl.item(), total_time / total_iters

ANS_RE = re.compile(r"(.*?)####\s*(-?[0-9.,]+)", re.DOTALL)  # Ensure re.DOTALL is used if multiline handling is necessary
INVALID_ANS = "[invalid]"

def extract_answer(completion):
    # Remove placeholder text enclosed in <<...>>
    completion = re.sub(r"<<.*?>>", "", completion)
    
    # Search for the pattern capturing text before and after '####'
    match = ANS_RE.search(completion)
    if match:
        # Extract rationale and answer, removing unnecessary spaces and commas
        rationale = match.group(1).strip()
        answer = match.group(2).strip().replace(",", "")  # Remove commas from numbers, if any
        return rationale, answer
    return INVALID_ANS  # Return invalid if no pattern matches

def evaluate_ppl(dataset_name, model, tokenizer, ctx_length, ignore_last=False):
	# max_length = model.seqlen 
	model_seqlen = ctx_length
	max_length = ctx_length
	stride = ctx_length

	if dataset_name == "wikitext":
		test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
		encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
		seq_len = encodings.input_ids.size(1)
	elif dataset_name == "ptb":
		testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
		encodings = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
		seq_len = encodings.input_ids.size(1)
	elif dataset_name == "c4":
		try:
			valdata = load_dataset(
				'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
			)
		except:
			print("Trying again but with a different config")
			valdata = load_dataset(
				'allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
			)
		encodings = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
		# encodings = encodings.input_ids[:, :(256 * model.seqlen)]
		seq_len = 256 * model_seqlen
	elif dataset_name == "gsm8k":
		testdata = load_dataset("gsm8k", "main", split='test')
		# code from get_gsm8k from data_gsm8k.py
		testenc = []
		for sample in testdata:
			question = sample['question']
			rationale, answer = extract_answer(sample['answer'])
			
			# question_rationale = question + '\nRationale: ' + rationale
			# question_rationale_enc = tokenizer(question_rationale, return_tensors='pt')
			# padded_question_rationale = tokenizer.pad(question_rationale_enc, max_length=seqlen, padding='max_length', truncation=True)
			
			# answer_enc = tokenizer(str(answer), return_tensors='pt')
			question = question + 'Answer this question:\n'
			question_en = tokenizer(question, return_tensors='pt')
			question_en = question_en.input_ids
			rationale_en = tokenizer(rationale, return_tensors='pt')
			rationale_en = rationale_en.input_ids
			# answer_en = tokenizer(answer, return_tensors='pt')
			# testloader.append((question_en, rationale_en, str(answer)))
			testenc.append((question_en, rationale_en, str(answer)))
		return eval_ppl_test_gsm8k(model, testenc)


	nlls = []
	prev_end_loc = 0
	total_time, total_iters = 0, 0
	for begin_loc in tqdm(range(0, seq_len, stride)):
		end_loc = min(begin_loc + max_length, seq_len)
		trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
		if (trg_len != stride) and ignore_last:
			break

		input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
		target_ids = input_ids.clone()
		target_ids[:, :-trg_len] = -100

		with torch.no_grad():
			start_ = time()
			outputs = model(input_ids, labels=target_ids)
			total_time += (time() - start_)
			total_iters += 1

			neg_log_likelihood = outputs.loss

		nlls.append(neg_log_likelihood)

		prev_end_loc = end_loc
		if end_loc == seq_len:
			break

	ppl = torch.exp(torch.stack(nlls).mean())
	return ppl.item(), total_time / total_iters