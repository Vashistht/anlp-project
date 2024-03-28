# %%
import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
import pdb
from time import time
import datetime
from lib.modelling_llama_mod import LlamaForCausalLM
from lib.eval import eval_ppl, eval_ppl_trainonly
from collections import defaultdict
import pickle as pkl
import random
from lib.scoring_model import ScoreModelHP
import wandb
from transformers.pytorch_utils import  find_pruneable_heads_and_indices, prune_linear_layer
import gc
import random

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

INF = 1e8

# %%
# # Prune the model according to the saved information
# def prune_model(model, tokenizer, prune_info_path):

# 	def get_param_count(model, exclude=['embed', 'head']):
# 		return sum([p.numel() for n, p in model.named_parameters() if not any(x in n for x in exclude)])

# 	epoch_ = 1
# 	mask_info_loc = os.path.join(prune_info_path, 'mask_info_{}.pkl'.format(epoch_))
# 	original_param_count = get_param_count(model)
# 	print('original model param count : {}'.format(original_param_count))
# 	while os.path.exists(mask_info_loc):
# 		with open(mask_info_loc, 'rb') as handle:
# 			mask_info = pkl.load(handle)

# 		for (name, module) in model.named_modules():
# 			if name not in mask_info: continue # We are not pruning this

# 			mask_ = mask_info[name]
# 			if name.endswith('mlp'):
# 				prune_mlp(mask_, module)
# 			elif name.endswith('self_attn'):
# 				prune_attn(mask_, module)
# 			else:
# 				raise ValueError("Invalid type found in mask_info : {}".format(name))

# 		gc.collect()
# 		torch.cuda.empty_cache() 
# 		print(f'epoch {epoch_}, param count is {get_param_count(model)}')
# 		epoch_ += 1
# 		mask_info_loc = os.path.join(prune_info_path, 'mask_info_{}.pkl'.format(epoch_))
# 	final_param_count = get_param_count(model)
# 	print('Final model sparsity is : {:.3f} '.format(1.0 - final_param_count/original_param_count))
# 	print('Final model param count : {}'.format(final_param_count))
# 	gc.collect()
# 	torch.cuda.empty_cache() 

# def prune_mlp(mask_, module):
# 	index = mask_.squeeze().nonzero().squeeze()
# 	new_gate_proj = (prune_linear_layer(module.gate_proj, index)).half()
# 	module.gate_proj = None
# 	module.gate_proj = new_gate_proj
# 	new_up_proj = (prune_linear_layer(module.up_proj, index)).half()
# 	module.up_proj  = None
# 	module.up_proj = new_up_proj
# 	new_down_proj = (prune_linear_layer(module.down_proj, index, dim=1)).half()
# 	module.down_proj = None
# 	module.down_proj = new_down_proj
# 	module.main_mask = None
# 	module.temp_mask = None
# 	module.intermed_cache = None
# 	module.intermediate_size = len(index)

# 	gc.collect()
# 	torch.cuda.empty_cache()

# def prune_attn(mask_, module):
# 	index = (mask_.squeeze() == 0).nonzero().squeeze()

# 	if index.numel() < 2:
# 		if index.numel() == 0: return # we are not pruning anything here
# 		index = [index]

# 	_, updated_indices = find_pruneable_heads_and_indices(
# 		index, module.num_heads, module.head_dim, set()
# 	)


# 	new_q_proj = (prune_linear_layer(module.q_proj, updated_indices)).half()
# 	module.q_proj = None
# 	module.q_proj = new_q_proj
	
# 	new_k_proj = (prune_linear_layer(module.k_proj, updated_indices)).half()
# 	module.k_proj = None
# 	module.k_proj = new_k_proj

# 	new_v_proj = (prune_linear_layer(module.v_proj, updated_indices)).half()
# 	module.v_proj = None
# 	module.v_proj = new_v_proj

# 	new_o_proj = (prune_linear_layer(module.o_proj, updated_indices, dim=1)).half()
# 	module.o_proj = None
# 	module.o_proj = new_o_proj

# 	module.num_heads = len(mask_.squeeze().nonzero())
# 	module.hidden_size = module.num_heads * module.head_dim

# 	module.main_mask = None
# 	module.temp_mask = None
# 	module.intermed_cache = None
# 	module.intermediate_size = module.num_heads

# 	gc.collect()
# 	torch.cuda.empty_cache() 

# def get_param_count(model, exclude=['embed', 'head']):
# 	return sum([p.numel() for n, p in model.named_parameters() if not any(x in n for x in exclude)])

# %%
def get_param_count(model, exclude=['embed', 'head']):
	return sum([p.numel() for n, p in model.named_parameters() if not any(x in n for x in exclude)])


# [NB] this pruning is specific to the LLaMA models. You will have to implement your own pruning for a custom model
def prune_mlp(mask_, module):
	# Reset pruning related information
	module.main_mask = None
	module.temp_mask = None
	module.intermed_cache = None
	module.ins_ = None

	if mask_.mean() == 0: # We are pruning the whole module here !
		print("We are pruning the whole mlp layer")
		module.gate_proj = None
		module.up_proj   = None
		module.down_proj = None
		module.intermediate_size = 0
		module.skip_computation = True
	else:
		index = mask_.squeeze().nonzero().squeeze()
		new_gate_proj = (prune_linear_layer(module.gate_proj, index)).half()
		module.gate_proj = None
		module.gate_proj = new_gate_proj
		new_up_proj = (prune_linear_layer(module.up_proj, index)).half()
		module.up_proj  = None
		module.up_proj = new_up_proj
		new_down_proj = (prune_linear_layer(module.down_proj, index, dim=1)).half()
		module.down_proj = None
		module.down_proj = new_down_proj
		module.intermediate_size = len(index)

	gc.collect()
	torch.cuda.empty_cache()

# [NB] this pruning is specific to the LLaMA models. You will have to implement your own pruning for a custom model
def prune_attn(mask_, module):

	module.main_mask = None
	module.temp_mask = None
	module.intermed_cache = None
	module.ins_ = None

	if mask_.mean() == 0: # We are pruning the whole module here !
		print('We are pruning a whole attention layer')
		module.q_proj = None
		module.k_proj = None
		module.v_proj = None
		module.o_proj = None
		module.skip_computation = True
		module.num_heads = 0
		module.hidden_size = 0
		module.intermediate_size = 0
	else:
		index = (mask_.squeeze() == 0).nonzero().squeeze()
		if index.numel() == 1:
			index = [index]

		_, updated_indices = find_pruneable_heads_and_indices(
			index, module.num_heads, module.head_dim, set()
		)

		new_q_proj = (prune_linear_layer(module.q_proj, updated_indices)).half()
		module.q_proj = None
		module.q_proj = new_q_proj

		new_k_proj = (prune_linear_layer(module.k_proj, updated_indices)).half()
		module.k_proj = None
		module.k_proj = new_k_proj

		new_v_proj = (prune_linear_layer(module.v_proj, updated_indices)).half()
		module.v_proj = None
		module.v_proj = new_v_proj

		new_o_proj = (prune_linear_layer(module.o_proj, updated_indices, dim=1)).half()
		module.o_proj = None
		module.o_proj = new_o_proj

		module.num_heads = len(mask_.squeeze().nonzero())
		module.hidden_size = module.num_heads * module.head_dim
		module.intermediate_size = module.num_heads


	gc.collect()
	torch.cuda.empty_cache() 

def prune_model(model, mask_info, tokenizer):
	for (name, module) in model.named_modules():
		if name not in mask_info: continue # We are not pruning this

		mask_ = mask_info[name]
		if name.endswith('mlp'):
			prune_mlp(mask_, module)
		elif name.endswith('self_attn'):
			prune_attn(mask_, module)
		else:
			raise ValueError("Invalid type found in mask_info : {}".format(name))

	gc.collect()
	torch.cuda.empty_cache() 

# %%
# For getting the LLM
def get_llm(model_name, cache_dir="llm_weights"):
	model = LlamaForCausalLM.from_pretrained(
		model_name, 
		torch_dtype=torch.float16, 
		cache_dir=cache_dir, 
		low_cpu_mem_usage=True, 
		device_map="auto"
	)
	model.seqlen = model.config.max_position_embeddings 
	if ('13b' in model_name) or ('65b' in model_name):
		model.seqlen = 2048 #Based on the values from the Lora-prune paper
	return model

# %%
# model_name_or_path = "meta-llama/Llama-2-7b-hf"
# config_name = "meta-llama/Llama-2-7b-hf"

model_name_or_path = 'princeton-nlp/Sheared-LLaMA-2.7B'
config_name = "princeton-nlp/Sheared-LLaMA-2.7B"
# %%
# torch.cuda.empty_cache()

# %%
model = get_llm(model_name= model_name_or_path)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
print('tokenizer done')

# %%
# Getting the initial evaluation of the model
# print('gsm8k')
# _, orig_test_ppl = eval_ppl(model, tokenizer, model.device, dataset='gsm8k')
# print('eval done original_test_ppl:', orig_test_ppl)

# print('wikitext2')
# _, orig_test_ppl = eval_ppl(model, tokenizer, model.device, dataset='wikitext2')
# print('eval done original_test_ppl:', orig_test_ppl)

# print('c4')
# _, orig_test_ppl = eval_ppl(model, tokenizer, model.device, dataset='c4')
# print('eval done original_test_ppl:', orig_test_ppl)

# %%
original_param_count = get_param_count(model)
print('original param count', original_param_count )

# %%
model.original_param_count = original_param_count
cur_sparsity = 1.0 - (get_param_count(model) / original_param_count)
epoch_ = 3

# %%
# save_loc =f"/home/vashistt/anlp-project/outdir/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2-wikitext_Adaptive=Yes/mask_info_{epoch_}.pkl"
save_loc =f"/home/vashistt/anlp-project/outdir_sheared_llama/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=200_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-sheared-llama2-wikitext_Adaptive=Yes/mask_info_{epoch_}.pkl"

# %%
# save_loc = os.path.join(args.save, 'mask_info_{}.pkl'.format(epoch_))
if os.path.exists(save_loc):
    print('Successfully loaded past pruning info')
    with open(save_loc, 'rb') as handle:
        mask_info = pkl.load(handle)
else:
    # mask_info = investigate_score_based_mask(args, model, wandb_run, epoch_=epoch_)
    # # Save the mask info for the epoch
    # with open(save_loc, 'wb') as handle:
    #     pkl.dump(mask_info, handle)
    print('not valid path')

print('Prune model')
prune_model(model, mask_info, tokenizer) # Do some stuffs here :)
cur_sparsity = 1.0 - (get_param_count(model) / original_param_count)
print(model)


# %%
# Evaluate the performance of the pruned model
# ppl_train, ppl_test = eval_ppl(model, tokenizer, model.device, dataset='gsm8k')

# ppl_train = 0
# print('Sparsity = {:.3f}| Train PPL = {:.3f} | Test PPL = {:.3f}'.format(cur_sparsity, ppl_train, ppl_test))

print('Sparsity = {:.3f}'.format(cur_sparsity))



print('wikitext2')
_, ppl_test = eval_ppl(model, tokenizer, model.device, dataset='wikitext2')
print('eval done test_ppl:', ppl_test)

# print('gsm8k')
# _, ppl_test = eval_ppl(model, tokenizer, model.device, dataset='gsm8k')
# print('eval done test_ppl:', ppl_test)


# print('c4')
# _, ppl_test = eval_ppl(model, tokenizer, model.device, dataset='c4')
# print('eval done test_ppl:', ppl_test)