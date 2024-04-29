## prune_target_epoch to get stats per epoch, other suppors to get get intermediate stats


#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""

"""
    Code here heavily burrows from https://github.com/locuslab/wanda/tree/main
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
from transformers.pytorch_utils import  find_pruneable_heads_and_indices, prune_linear_layer
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List 
import pdb
import pickle as pkl
import gc
import time
from peft import PeftModel
# Uncomment this out if running the Eleuther evaluation harness
# import lm_eval
from lm_eval import evaluator

import datasets
import evaluate
import torch
from datasets import load_dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from evaluate_ppl import evaluate_ppl

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.29.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    lora_r: Optional[int] = field(
        default=64,
        metadata={"help": "parameter lora_r"},
    )
    lora_alpha_ratio: Optional[float] = field(
        default=2.0,
        metadata={"help": "parameter lora_alpha"},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": "parameter lora_dropout"},
    )

    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    prune_info_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    
    add_finetuned_adapter: bool = field(
        default=False,
        metadata={"help": "Whether to add a finetuned adapter to the model"},
    )
    
    prune_target_epoch : Optional[int] = field(
        default=3,
        metadata={"help": "The number of masks we have for the model in the prune_info_path"},
    )
        
    do_eleuther_eval: bool = field(
        default=True, metadata={"help": "Whether to run the Eleuther Evaluation Harness"}
    )

    do_eleuther_eval_og_model: bool = field(
        default=False, metadata={"help": "Whether to run the Eleuther Evaluation Harness for the original model"}
    )
    
    full_ft: bool = field(
        default=False, metadata={"help": "Whether to perform full fine-tuning on the model"}
    )
    
    kl_weight: float = field(
        default=0.01, metadata={"help": "The weight to put on the kl term"}
    )
    
    hidden_mse_weight: float = field(
        default=0.01, metadata={"help": "The weight to put on the kl term"}
    )

    ctx_length: Optional[int] = field(
        default=2048,
        metadata={"help": "context length"},
    )


    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default="wikitext", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default="wikitext-2-raw-v1", metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


class CustomTrainer(Trainer):
    def set_distill_info(self, teacher_model, kl_weight=1.0, hidden_mse_weight=1.0):
        teacher_model.eval()
        self.kl_weight = kl_weight
        self.hidden_mse_weight = hidden_mse_weight
        self.kl_fnct = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.mse_fnct = torch.nn.MSELoss()
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs['output_hidden_states'] = True
        with torch.no_grad():
            teacher_out = self.teacher_model(**inputs)
            teacher_logits = torch.nn.functional.log_softmax(teacher_out["logits"], dim=-1)
            teacher_hidden = teacher_out['hidden_states']

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # calculate the classic distillation loss
        student_logits = torch.nn.functional.log_softmax(outputs["logits"], dim=-1)
        kl_loss = self.kl_fnct(student_logits, teacher_logits)

        # now calculate the per-layer hidden mse loss
        hidden_loss = 0.0
        for id_ in range(1, len(teacher_hidden)):
            teacher_h = teacher_hidden[id_]
            student_h = outputs['hidden_states'][id_]
            hidden_loss += self.mse_fnct(teacher_h, student_h)

        loss = outputs['loss']
        loss = loss + self.kl_weight*kl_loss + self.hidden_mse_weight*hidden_loss

        return (loss, outputs) if return_outputs else loss



# Prune the model according to the saved information
def prune_model(model, tokenizer, prune_info_path, target_epoch=3):

    def get_param_count(model, exclude=['embed', 'head']):
        return sum([p.numel() for n, p in model.named_parameters() if not any(x in n for x in exclude)])

    epoch_ = 1
    mask_info_loc = os.path.join(prune_info_path, 'mask_info_{}.pkl'.format(epoch_))
    original_param_count = get_param_count(model)
    print('STDOUT: original model param count : {}'.format(original_param_count))
    
    while (os.path.exists(mask_info_loc)) and (epoch_ <= target_epoch):
        print('STDOUT: Pruning for epoch : {}'.format(epoch_) )
        with open(mask_info_loc, 'rb') as handle:
            mask_info = pkl.load(handle)

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
        print(f'STDOUT: epoch {epoch_}, param count is {get_param_count(model)}')
        epoch_ += 1
        mask_info_loc = os.path.join(prune_info_path, 'mask_info_{}.pkl'.format(epoch_))
    final_param_count = get_param_count(model)
    print('STDOUT: Final model sparsity is : {:.3f} '.format(1.0 - final_param_count/original_param_count))
    print('STDOUT: Final model param count : {}'.format(final_param_count))
    gc.collect()
    torch.cuda.empty_cache() 

def prune_mlp(mask_, module):
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
    module.main_mask = None
    module.temp_mask = None
    module.intermed_cache = None
    module.intermediate_size = len(index)

    gc.collect()
    torch.cuda.empty_cache()

def prune_attn(mask_, module):
    index = (mask_.squeeze() == 0).nonzero().squeeze()

    if index.numel() < 2:
        if index.numel() == 0: return # we are not pruning anything here
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

    module.main_mask = None
    module.temp_mask = None
    module.intermed_cache = None
    module.intermediate_size = module.num_heads

    gc.collect()
    torch.cuda.empty_cache() 

def get_param_count(model, exclude=['embed', 'head']):
    return sum([p.numel() for n, p in model.named_parameters() if not any(x in n for x in exclude)])

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    # TODO[ldery] -- we need to change this to account for fine-tuning on the right dataset
    # raw_datasets = load_dataset(
    #     'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz', 'validation': 'en/c4-validation.00000-of-00008.json.gz'}
    # )

    # if "c4" not in data_args.dataset_name:
    #     raw_datasets["validation"] = load_dataset(
    #         data_args.dataset_name,
    #         data_args.dataset_config_name,
    #         split=f"train[:{data_args.validation_split_percentage}%]",
    #         use_auth_token=True if model_args.use_auth_token else None,
    #         streaming=data_args.streaming,
    #     )
    #     raw_datasets["train"] = load_dataset(
    #         data_args.dataset_name,
    #         data_args.dataset_config_name,
    #         split=f"train[{data_args.validation_split_percentage}%:]",
    #         use_auth_token=True if model_args.use_auth_token else None,
    #         streaming=data_args.streaming,
    #     )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs, trust_remote_code=True)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        # "use_fast": model_args.use_fast_tokenizer,
        "use_fast": False,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_args.config_name, use_fast=False, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.float16, cache_dir=model_args.cache_dir, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True)
    model.seqlen = model.config.max_position_embeddings

    '''
    # outout_txt
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "output_gsm8k_5shot.txt")
    print('File path: ', file_path)

    try:
        with open(file_path, 'w') as file:
            print("Num params = : ", get_param_count(model))
            file.write("Num params = : {}\n".format(get_param_count(model)))
            # file.flush()
        print("File created successfully.")
    except IOError as e:
        print("Error creating the file:", e)
        
    '''
    # Do the pre-training evaluation
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        model.eval()
        print(f'STDOUT: Evaluating on {data_args.dataset_name}')
        og_ppl, og_runtime = evaluate_ppl(data_args.dataset_name, model, tokenizer, model.seqlen)
        out_str = "STDOUT: Original perplexity on {} = {:.3f}".format(data_args.dataset_name, og_ppl)
        print(out_str)

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")


    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()
        return result

    should_i_do_eleuther_eval_og = model_args.do_eleuther_eval_og_model
    print('STDOUT: Eleuther eval for og model: ', should_i_do_eleuther_eval_og)
    og_params = get_param_count(model)
    print("STDOUT: Num params = : ",  get_param_count(model))

    if should_i_do_eleuther_eval_og:
        print('STDOUT: eleuther eval for original model')
    # 	# transformers.modeling_utils.load_sharded_checkpoint(model, training_args.output_dir)
        results = evaluator.simple_evaluate(
            model="hf-causal-experimental",
            model_args="pretrained={}".format(model_args.model_name_or_path),
            tasks=["winogrande", "boolq", "arc_challenge", "arc_easy", "hellaswag"], # main one here
            # tasks = ['gsm8k'],
            num_fewshot=0,
            limit = .005, # how much of the original dataset to test on 
            no_cache=True,
            pretrained_model=model,
            write_out=True, 
            output_base_path='original-model_all_ds',  # writes to the current dir if commented out 
        )
        updated_results = {'results': results['results']}
        print('STDOUT:', updated_results)
        results_str = 'orginal-model \n' + str(updated_results)
        # out_file.write(results_str + "\n")

    
    
    
    if model_args.prune_info_path is not None:
        prune_model(model, tokenizer, model_args.prune_info_path, target_epoch= model_args.prune_target_epoch)
        print("STDOUT: Num params = : ", get_param_count(model))
        final_sparsity = 1.0 - get_param_count(model)/og_params
        print('STDOUT: Final sparsity is : {:.3f}'.format(final_sparsity))
        
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("*** Evaluate ***")
        model.eval()
        start_time = time.time()
        before_train_ppl, final_runtime = evaluate_ppl(data_args.dataset_name, model, tokenizer, model.seqlen)
        speedup = og_runtime / final_runtime
        out_str = "STDOUT: [Dataset: {}| SpeedUp={:.3f}] Original perplexity = {:.3f} | Before Training perplexity = {:.3f}".format(data_args.dataset_name,speedup, og_ppl, before_train_ppl, speedup)
        # out_file.write(out_str + "\n")
        print('STDOUT: ', out_str)

        should_i_do_eleuther_eval = model_args.do_eleuther_eval
        print(f'STDOUT: Eleuther eval for pruned model (no finetuning): {should_i_do_eleuther_eval}')
        
        if should_i_do_eleuther_eval:
            #transformers.modeling_utils.load_sharded_checkpoint(model, training_args.output_dir)
            results = evaluator.simple_evaluate(
                model="hf-causal-experimental",
                model_args="pretrained={}".format(model_args.model_name_or_path),
                tasks=["winogrande", "boolq", "arc_challenge", "arc_easy", "hellaswag"], # main one here
                # tasks = ['gsm8k'],
                num_fewshot=0,
                limit = .005, # how much of the original dataset to test on 
                # num_fewshot={"hellaswag": 0, "arc_challenge":0}
                no_cache=True,
                pretrained_model=model,
                write_out=True, 
                output_base_path=f'pruned_sparsity-{final_sparsity:.3f}_all-dataset', # writes to the current dir
            )
            updated_results = {'results': results['results']}
            print('STDOUT:', updated_results)
    
           
    finetuned_model = model_args.add_finetuned_adapter
    print('STDOUT: Finetuning the Model: ', finetuned_model)
    # if model_args.add_finetuned_adapter, meaning the adapters for finetuning are there and we want to finetune
    if finetuned_model:
        adapter_name = '/home/vashistt/Desktop/anlp-project/finetuned_model_prune_c4_ft_wiki/'
        model = PeftModel.from_pretrained(model, adapter_name, adapter_name="prune_c4_ft_wiki_adapter")

        if should_i_do_eleuther_eval:
            #transformers.modeling_utils.load_sharded_checkpoint(model, training_args.output_dir)
            print('STDOUT: Eleuther eval for pruned model + finetuning')
            results = evaluator.simple_evaluate(
                model="hf-causal-experimental",
                model_args="pretrained={}".format(model_args.model_name_or_path),
                tasks=["winogrande", "boolq", "arc_challenge", "arc_easy", "hellaswag"], # main one here
                # tasks = ['gsm8k'],
                num_fewshot=0,
                limit = .005, # how much of the original dataset to test on 
                # num_fewshot={"hellaswag": 0, "arc_challenge":0}
                no_cache=True,
                pretrained_model=model,
                write_out=True, 
                output_base_path=f'finetuned_sparsity-{final_sparsity:.3f}_all_dataset', # writes to the current dir
            )
            
        updated_results = {'results': results['results']}
        print('STDOUT:', updated_results)
        
    # with open(file_path, 'w') as out_file:
    #     out_file.write("\n")
    #     out_file.write(results_str + "\n")
    #     out_file.flush()

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

    
#     results = evaluator.simple_evaluate(
#     model="hf-causal-experimental",
#     model_args="pretrained={}".format(model_args.model_name_or_path),
#     tasks=["winogrande", "boolq", "arc_challenge", "arc_easy", "hellaswag"], # main one here
#     # tasks = ['gsm8k'],
#     num_fewshot=0,
#     limit = .01, # how much of the original dataset to test on 
#     no_cache=True,
#     pretrained_model=model,
#     write_out=True, 
#     output_base_path='/Users/vashisth/Documents/GitHub/ANLP_projects/anlp-project/logs_errors_outputs/gsm8k-pruned/finetuned-all-ds.json', # writes to the current dir if commented out 
# )