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
from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
import gc
import random

# wandb login in bash


print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

INF = 1e8

def get_param_count(model, exclude=['embed', 'head']):
    return sum([p.numel() for n, p in model.named_parameters() if not any(x in n for x in exclude)])

def prune_mlp(mask_, module):
    module.main_mask = None
    module.temp_mask = None
    module.intermed_cache = None
    module.ins_ = None

    if mask_.mean() == 0:  # We are pruning the whole module here !
        print("We are pruning the whole mlp layer")
        module.gate_proj = None
        module.up_proj = None
        module.down_proj = None
        module.intermediate_size = 0
        module.skip_computation = True
    else:
        index = mask_.squeeze().nonzero().squeeze()
        new_gate_proj = (prune_linear_layer(module.gate_proj, index)).half()
        module.gate_proj = None
        module.gate_proj = new_gate_proj
        new_up_proj = (prune_linear_layer(module.up_proj, index)).half()
        module.up_proj = None
        module.up_proj = new_up_proj
        new_down_proj = (prune_linear_layer(module.down_proj, index, dim=1)).half()
        module.down_proj = None
        module.down_proj = new_down_proj
        module.intermediate_size = len(index)

    gc.collect()
    torch.cuda.empty_cache()

def prune_attn(mask_, module):
    module.main_mask = None
    module.temp_mask = None
    module.intermed_cache = None
    module.ins_ = None

    if mask_.mean() == 0:  # We are pruning the whole module here !
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
        if name not in mask_info:
            continue  # We are not pruning this

        mask_ = mask_info[name]
        if name.endswith('mlp'):
            prune_mlp(mask_, module)
        elif name.endswith('self_attn'):
            prune_attn(mask_, module)
        else:
            raise ValueError("Invalid type found in mask_info : {}".format(name))

    gc.collect()
    torch.cuda.empty_cache()

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
        model.seqlen = 2048  # Based on the values from the Lora-prune paper
    return model

# wanb_name = 'LLaMA-2-7B-hf'
# model_name_or_path = 'meta-llama/Llama-2-7b-hf'
# config_name = 'meta-llama/Llama-2-7b-hf'
# mask_dir = "/home/vashistt/anlp-project/outdir_llama_2_7b/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2-wikitext_Adaptive=Yes"

wanb_name = 'Sheared-LLaMA-2.7B'
model_name_or_path = 'princeton-nlp/Sheared-LLaMA-2.7B'
config_name = "princeton-nlp/Sheared-LLaMA-2.7B"
mask_dir = "/home/vashistt/anlp-project/outdir_sheared_llama/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=200_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-sheared-llama2-wikitext_Adaptive=Yes"

wandb_run = wandb.init(project=f'{wanb_name}-eval-table3', config={'epochs': 3})


model = get_llm(model_name=model_name_or_path)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
print('tokenizer done')
# # %%
# model = get_llm(model_name= model_name_or_path)
# model.eval()
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
# print('tokenizer done')

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
epoch_ = 1

# original_param_count = get_param_count(model)
# print('original param count', original_param_count)
# model.original_param_count = original_param_count

# # Log 0 sparsity performance
# cur_sparsity = 0.0
# wandb_run.log({'sparsity': cur_sparsity})

# print('gsm8k')
# _, orig_test_ppl = eval_ppl(model, tokenizer, model.device, dataset='gsm8k')
# print('eval done original_test_ppl:', orig_test_ppl)
# wandb_run.log({'gsm8k_ppl': orig_test_ppl})

# print('wikitext2')
# _, orig_test_ppl = eval_ppl(model, tokenizer, model.device, dataset='wikitext2')
# print('eval done original_test_ppl:', orig_test_ppl)
# wandb_run.log({'wikitext2_ppl': orig_test_ppl})

# print('c4')
# _, orig_test_ppl = eval_ppl(model, tokenizer, model.device, dataset='c4')
# print('eval done original_test_ppl:', orig_test_ppl)
# wandb_run.log({'c4_ppl': orig_test_ppl})

# print('boolq')
# _, orig_test_ppl = eval_ppl(model, tokenizer, model.device, dataset='boolq')
# print('eval done original_test_ppl:', orig_test_ppl)
# wandb_run.log({'boolq_ppl': orig_test_ppl})

# print('hellaswag')
# _, orig_test_ppl = eval_ppl(model, tokenizer, model.device, dataset='hellaswag')
# print('eval done original_test_ppl:', orig_test_ppl)
# wandb_run.log({'hellaswag_ppl': orig_test_ppl})

# print('winogrande')
# _, orig_test_ppl = eval_ppl(model, tokenizer, model.device, dataset='winogrande')
# print('eval done original_test_ppl:', orig_test_ppl)
# wandb_run.log({'winogrande_ppl': orig_test_ppl})

# print('arc-e')
# _, orig_test_ppl = eval_ppl(model, tokenizer, model.device, dataset='arc-e')
# print('eval done original_test_ppl:', orig_test_ppl)
# wandb_run.log({'arc-e_ppl': orig_test_ppl})

# print('arc-c')
# _, orig_test_ppl = eval_ppl(model, tokenizer, model.device, dataset='arc-c')
# print('eval done original_test_ppl:', orig_test_ppl)
# wandb_run.log({'arc-c_ppl': orig_test_ppl})

epoch_ = 1

while True:
    save_loc = f"{mask_dir}/mask_info_{epoch_}.pkl"

    if os.path.exists(save_loc):
        print('Successfully loaded past pruning info')
        with open(save_loc, 'rb') as handle:
            mask_info = pkl.load(handle)
    else:
        print('not valid path')
        break

    print('Prune model')
    prune_model(model, mask_info, tokenizer)
    cur_sparsity = 1.0 - (get_param_count(model) / original_param_count)
    print(model)

    wandb_run.log({'sparsity': cur_sparsity})

    # print('gsm8k')
    # _, ppl_test = eval_ppl(model, tokenizer, model.device, dataset='gsm8k')
    # print('eval done test_ppl:', ppl_test)
    # wandb_run.log({'gsm8k_ppl': ppl_test})

    # print('wikitext2')
    # _, ppl_test = eval_ppl(model, tokenizer, model.device, dataset='wikitext2')
    # print('eval done test_ppl:', ppl_test)
    # wandb_run.log({'wikitext2_ppl': ppl_test})

    # print('c4')
    # _, ppl_test = eval_ppl(model, tokenizer, model.device, dataset='c4')
    # print('eval done test_ppl:', ppl_test)
    # wandb_run.log({'c4_ppl': ppl_test})

    print('boolq')
    _, ppl_test = eval_ppl(model, tokenizer, model.device, dataset='boolq')
    print('eval done test_ppl:', ppl_test)
    wandb_run.log({'boolq_ppl': ppl_test})

    print('hellaswag')
    _, ppl_test = eval_ppl(model, tokenizer, model.device, dataset='hellaswag')
    print('eval done test_ppl:', ppl_test)
    wandb_run.log({'hellaswag_ppl': ppl_test})

    print('winogrande')
    _, ppl_test = eval_ppl(model, tokenizer, model.device, dataset='winogrande')
    print('eval done test_ppl:', ppl_test)
    wandb_run.log({'winogrande_ppl': ppl_test})

    print('arc-e')
    _, ppl_test = eval_ppl(model, tokenizer, model.device, dataset='arc-e')
    print('eval done test_ppl:', ppl_test)
    wandb_run.log({'arc-e_ppl': ppl_test})

    print('arc-c')
    _, ppl_test = eval_ppl(model, tokenizer, model.device, dataset='arc-c')
    print('eval done test_ppl:', ppl_test)
    wandb_run.log({'arc-c_ppl': ppl_test})

    epoch_ += 1
