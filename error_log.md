# Error Log

## Run 0: setting up [Sun Mar 24]
Device: ml.g4dn.xlarge


    ```OSError: [Errno 28] No space left on device
    We hit a nan or inf. Stopping
    We hit a nan or inf. Resettinig 
    We hit a nan or inf. Stopping
    wandb: ERROR Internal wandb error: file data was not synced
    '''

## Run: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
Device: ml.g4dn.2xlarge

    '''
    Traceback (most recent call last):
      File "/home/ec2-user/SageMaker/anlp-project/main.py", line 603, in <module>
    main()
      File "/home/ec2-user/SageMaker/anlp-project/main.py", line 550, in main
        model = get_llm(args.model, args.cache_dir)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/SageMaker/anlp-project/main.py", line 109, in get_llm
        model = LlamaForCausalLM.from_pretrained(
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.11/site-packages/transformers/modeling_utils.py", line 3531, in from_pretrained
        ) = cls._load_pretrained_model(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.11/site-packages/transformers/modeling_utils.py", line 3958, in _load_pretrained_model
        new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
                                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.11/site-packages/transformers/modeling_utils.py", line 812, in _load_state_dict_into_meta_model
        set_module_tensor_to_device(model, param_name, param_device, **set_module_kwargs)
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.11/site-packages/accelerate/utils/modeling.py", line 348, in set_module_tensor_to_device
        raise ValueError(
    ValueError: Trying to set a tensor of shape torch.Size([256, 2048]) in "weight" (which has shape torch.Size([2048, 2048])), this look incorrect.
    '''
    
## Running with phi 1.5

    '''
    You are using a model of type phi to instantiate a model of type llama. This is not supported for all configurations of models and can yield errors.'''


## Sheared LLama 

(changed seed_ = random.randint(0, 1e4) to 
seed_ = random.randint(0, int(1e4) )

    '''
    eval done original_test_ppl: 6.4152679443359375
    current sparsity 0.0
    Gathering statistics for pruning
    evaluating on wikitext2
    nsamples 8
    sample 0
    Traceback (most recent call last):
      File "/home/ec2-user/SageMaker/anlp-project/main.py", line 605, in <module>
        main()
      File "/home/ec2-user/SageMaker/anlp-project/main.py", line 583, in main
        mask_info = investigate_score_based_mask(args, model, wandb_run, epoch_=epoch_)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/SageMaker/anlp-project/main.py", line 302, in investigate_score_based_mask
        score_info = get_random_mask_scores(
                     ^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/SageMaker/anlp-project/main.py", line 65, in get_random_mask_scores
        seed_ = random.randint(0, 1e4)
                ^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/random.py", line 336, in randint
        return self.randrange(a, b+1)
               ^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/random.py", line 312, in randrange
        istop = _index(stop)
                ^^^^^^^^^^^^
    TypeError: 'float' object cannot be interpreted as an integer
    '''


## Sheared LLama

'''
eval done original_test_ppl: 6.4152679443359375
current sparsity 0.0
Gathering statistics for pruning
evaluating on wikitext2
nsamples 8
sample 0
Traceback (most recent call last):
  File "/home/ec2-user/SageMaker/anlp-project/main.py", line 605, in <module>
    main()
  File "/home/ec2-user/SageMaker/anlp-project/main.py", line 583, in main
    mask_info = investigate_score_based_mask(args, model, wandb_run, epoch_=epoch_)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ec2-user/SageMaker/anlp-project/main.py", line 302, in investigate_score_based_mask
    score_info = get_random_mask_scores(
                 ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ec2-user/SageMaker/anlp-project/main.py", line 65, in get_random_mask_scores
    seed_ = random.randint(0, 1e4)
            ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/random.py", line 336, in randint
    return self.randrange(a, b+1)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/random.py", line 312, in randrange
    istop = _index(stop)
            ^^^^^^^^^^^^
TypeError: 'float' object cannot be interpreted as an integer
'''

## Run: Sheared LLama Finetuning

    '''
    Traceback (most recent call last):
      File "/home/ec2-user/SageMaker/anlp-project/lora_ft/finetune_lm.py", line 853, in <module>
        main()
      File "/home/ec2-user/SageMaker/anlp-project/lora_ft/finetune_lm.py", line 663, in main
        before_train_ppl, final_runtime = evaluate_ppl(data_args.dataset_name, model, tokenizer, model.seqlen)
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/SageMaker/anlp-project/lora_ft/evaluate_ppl.py", line 62, in evaluate_ppl
        outputs = model(input_ids, labels=target_ids)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
        return forward_call(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py", line 1196, in forward
        outputs = self.model(
                  ^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
        return forward_call(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py", line 1016, in forward
        layer_outputs = decoder_layer(
                        ^^^^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
        return forward_call(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py", line 739, in forward
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
                                                              ^^^^^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
        return forward_call(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py", line 641, in forward
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    RuntimeError: shape '[1, 4096, 20, 128]' is invalid for input of size 5242880 
    '''

# Finetuning LLama 

  ```
  (prune_llm) [vashistt@lovelace lora_ft]$ CUDA_VISIBLE_DEVICES=8 python3 finetune_lm.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512        --lora_r 128    --learning_rate 1e-4            --lora_alpha_ratio 4    --per_device_train_batch_size 1         --per_device_eval_batch_size 8       --do_train      --do_eval       --max_train_samples 15000       --max_eval_samples 128  --overwrite_output_dir  --output_dir "${output_dir}"    --prune_info_path "${location}/pruned_model.pkl"     --hidden_mse_weight 0.0         --kl_weight 0.01        --dataset_name "wikitext" 
  03/27/2024 21:44:45 - WARNING - __main__ - Process rank: 0, device: cuda:0, n_gpu: 1distributed training: True, 16-bits training: False
  03/27/2024 21:44:45 - INFO - __main__ - Training/evaluation parameters TrainingArguments(
  _n_gpu=1,
  accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True},
  adafactor=False,
  adam_beta1=0.9,
  adam_beta2=0.999,
  adam_epsilon=1e-08,
  auto_find_batch_size=False,
  bf16=False,
  bf16_full_eval=False,
  data_seed=None,
  dataloader_drop_last=False,
  dataloader_num_workers=0,
  dataloader_persistent_workers=False,
  dataloader_pin_memory=True,
  dataloader_prefetch_factor=None,
  ddp_backend=None,
  ddp_broadcast_buffers=None,
  ddp_bucket_cap_mb=None,
  ddp_find_unused_parameters=None,
  ddp_timeout=1800,
  debug=[],
  deepspeed=None,
  disable_tqdm=False,
  dispatch_batches=None,
  do_eval=True,
  do_predict=False,
  do_train=True,
  eval_accumulation_steps=None,
  eval_delay=0,
  eval_steps=None,
  evaluation_strategy=IntervalStrategy.NO,
  fp16=False,
  fp16_backend=auto,
  fp16_full_eval=False,
  fp16_opt_level=O1,
  fsdp=[],
  fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False},
  fsdp_min_num_params=0,
  fsdp_transformer_layer_cls_to_wrap=None,
  full_determinism=False,
  gradient_accumulation_steps=1,
  gradient_checkpointing=False,
  gradient_checkpointing_kwargs=None,
  greater_is_better=None,
  group_by_length=False,
  half_precision_backend=auto,
  hub_always_push=False,
  hub_model_id=None,
  hub_private_repo=False,
  hub_strategy=HubStrategy.EVERY_SAVE,
  hub_token=<HUB_TOKEN>,
  ignore_data_skip=False,
  include_inputs_for_metrics=False,
  include_num_input_tokens_seen=False,
  include_tokens_per_second=False,
  jit_mode_eval=False,
  label_names=None,
  label_smoothing_factor=0.0,
  learning_rate=0.0001,
  length_column_name=length,
  load_best_model_at_end=False,
  local_rank=0,
  log_level=passive,
  log_level_replica=warning,
  log_on_each_node=True,
  logging_dir=/home/vashistt/anlp-project/finetuned_model/runs/Mar27_21-44-45_lovelace.ece.local.cmu.edu,
  logging_first_step=False,
  logging_nan_inf_filter=True,
  logging_steps=500,
  logging_strategy=IntervalStrategy.STEPS,
  lr_scheduler_kwargs={},
  lr_scheduler_type=SchedulerType.LINEAR,
  max_grad_norm=1.0,
  max_steps=-1,
  metric_for_best_model=None,
  mp_parameters=,
  neftune_noise_alpha=None,
  no_cuda=False,
  num_train_epochs=1.0,
  optim=OptimizerNames.ADAMW_TORCH,
  optim_args=None,
  optim_target_modules=None,
  output_dir=/home/vashistt/anlp-project/finetuned_model,
  overwrite_output_dir=True,
  past_index=-1,
  per_device_eval_batch_size=8,
  per_device_train_batch_size=1,
  prediction_loss_only=False,
  push_to_hub=False,
  push_to_hub_model_id=None,
  push_to_hub_organization=None,
  push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
  ray_scope=last,
  remove_unused_columns=True,
  report_to=['wandb'],
  resume_from_checkpoint=None,
  run_name=/home/vashistt/anlp-project/finetuned_model,
  save_on_each_node=False,
  save_only_model=False,
  save_safetensors=True,
  save_steps=500,
  save_strategy=IntervalStrategy.STEPS,
  save_total_limit=None,
  seed=42,
  skip_memory_metrics=True,
  split_batches=None,
  tf32=None,
  torch_compile=False,
  torch_compile_backend=None,
  torch_compile_mode=None,
  torchdynamo=None,
  tpu_metrics_debug=False,
  tpu_num_cores=None,
  use_cpu=False,
  use_ipex=False,
  use_legacy_prediction_loop=False,
  use_mps_device=False,
  warmup_ratio=0.0,
  warmup_steps=0,
  weight_decay=0.0,
  )
  Using custom data configuration default-a3e66ef7800043cd
  03/27/2024 21:44:52 - INFO - datasets.builder - Using custom data configuration default-a3e66ef7800043cd
  Loading Dataset Infos from /home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/datasets/packaged_modules/json
  03/27/2024 21:44:52 - INFO - datasets.info - Loading Dataset Infos from /home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/datasets/packaged_modules/json
  Overwrite dataset info from restored data version if exists.
  03/27/2024 21:44:52 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
  03/27/2024 21:44:52 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
  Found cached dataset c4 (/home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2)
  03/27/2024 21:44:52 - INFO - datasets.builder - Found cached dataset c4 (/home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
  03/27/2024 21:44:52 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
  /home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/datasets/load.py:2516: FutureWarning: 'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.
  You can remove this warning by passing 'token=<use_auth_token>' instead.
    warnings.warn(
  Overwrite dataset info from restored data version if exists.
  03/27/2024 21:44:55 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/27/2024 21:44:55 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  03/27/2024 21:44:55 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/27/2024 21:44:55 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  Overwrite dataset info from restored data version if exists.
  03/27/2024 21:44:58 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/27/2024 21:44:58 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  03/27/2024 21:44:58 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/27/2024 21:44:58 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  [INFO|configuration_utils.py:726] 2024-03-27 21:44:58,231 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/config.json
  [INFO|configuration_utils.py:789] 2024-03-27 21:44:58,232 >> Model config LlamaConfig {
    "_name_or_path": "meta-llama/Llama-2-7b-hf",
    "architectures": [
      "LlamaForCausalLM"
    ],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 4096,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 32,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "rope_theta": 10000.0,
    "tie_word_embeddings": false,
    "torch_dtype": "float16",
    "transformers_version": "4.39.1",
    "use_cache": true,
    "vocab_size": 32000
  }

  [INFO|tokenization_utils_base.py:2084] 2024-03-27 21:44:58,276 >> loading file tokenizer.model from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer.model
  [INFO|tokenization_utils_base.py:2084] 2024-03-27 21:44:58,276 >> loading file added_tokens.json from cache at None
  [INFO|tokenization_utils_base.py:2084] 2024-03-27 21:44:58,276 >> loading file special_tokens_map.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/special_tokens_map.json
  [INFO|tokenization_utils_base.py:2084] 2024-03-27 21:44:58,276 >> loading file tokenizer_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer_config.json
  [INFO|tokenization_utils_base.py:2084] 2024-03-27 21:44:58,276 >> loading file tokenizer.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer.json
  [INFO|configuration_utils.py:726] 2024-03-27 21:44:58,408 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/config.json
  [INFO|configuration_utils.py:789] 2024-03-27 21:44:58,409 >> Model config LlamaConfig {
    "_name_or_path": "meta-llama/Llama-2-7b-hf",
    "architectures": [
      "LlamaForCausalLM"
    ],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 4096,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 32,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "rope_theta": 10000.0,
    "tie_word_embeddings": false,
    "torch_dtype": "float16",
    "transformers_version": "4.39.1",
    "use_cache": true,
    "vocab_size": 32000
  }

  [INFO|modeling_utils.py:3283] 2024-03-27 21:44:58,412 >> loading weights file model.safetensors from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/model.safetensors.index.json
  [INFO|modeling_utils.py:1417] 2024-03-27 21:44:58,412 >> Instantiating LlamaForCausalLM model under default dtype torch.float16.
  [INFO|configuration_utils.py:928] 2024-03-27 21:44:58,413 >> Generate config GenerationConfig {
    "bos_token_id": 1,
    "eos_token_id": 2
  }

  Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.27s/it]
  [INFO|modeling_utils.py:4024] 2024-03-27 21:45:01,635 >> All model checkpoint weights were used when initializing LlamaForCausalLM.

  [INFO|modeling_utils.py:4032] 2024-03-27 21:45:01,636 >> All the weights of LlamaForCausalLM were initialized from the model checkpoint at meta-llama/Llama-2-7b-hf.
  If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
  [INFO|configuration_utils.py:883] 2024-03-27 21:45:01,763 >> loading configuration file generation_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/generation_config.json
  [INFO|configuration_utils.py:928] 2024-03-27 21:45:01,763 >> Generate config GenerationConfig {
    "bos_token_id": 1,
    "do_sample": true,
    "eos_token_id": 2,
    "max_length": 4096,
    "pad_token_id": 0,
    "temperature": 0.6,
    "top_p": 0.9
  }

  Num params = :  6476271616
  03/27/2024 21:45:01 - INFO - __main__ - *** Evaluate ***
  Overwrite dataset info from restored data version if exists.
  03/27/2024 21:45:04 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/27/2024 21:45:04 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  03/27/2024 21:45:04 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/27/2024 21:45:04 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  99%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏ | 83/84 [00:47<00:00,  1.76it/s]
  Original perplexity on wikitext = 5.110
  Loading cached processed dataset at /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-d03aa48fc602ceae.arrow
  03/27/2024 21:45:55 - INFO - datasets.arrow_dataset - Loading cached processed dataset at /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-d03aa48fc602ceae.arrow
  Loading cached processed dataset at /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-6cce13c6ba039ed5.arrow
  03/27/2024 21:45:55 - INFO - datasets.arrow_dataset - Loading cached processed dataset at /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-6cce13c6ba039ed5.arrow
  Loading cached processed dataset at /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-7925abbcc4bb6061.arrow
  03/27/2024 21:45:55 - INFO - datasets.arrow_dataset - Loading cached processed dataset at /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-7925abbcc4bb6061.arrow
  Loading cached processed dataset at /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-85fc31e9d4c09ef5.arrow
  03/27/2024 21:45:55 - INFO - datasets.arrow_dataset - Loading cached processed dataset at /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/cache-85fc31e9d4c09ef5.arrow
  Final model sparsity is : 0.000 
  Num params = :  6476271616
  03/27/2024 21:45:55 - INFO - __main__ - *** Evaluate ***
  Overwrite dataset info from restored data version if exists.
  03/27/2024 21:45:58 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/27/2024 21:45:58 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  03/27/2024 21:45:58 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/27/2024 21:45:58 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  99%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏ | 83/84 [00:47<00:00,  1.76it/s]
  [SpeedUp=1.043] Original perplexity on wikitext = 5.110 | Before Training perplexity on wikitext = 5.110
  /home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/peft/utils/other.py:136: FutureWarning: prepare_model_for_int8_training is deprecated and will be removed in a future version. Use prepare_model_for_kbit_training instead.
    warnings.warn(
  /home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/transformers/utils/import_utils.py:519: FutureWarning: `is_torch_tpu_available` is deprecated and will be removed in 4.41.0. Please use the `is_torch_xla_available` instead.
    warnings.warn(
  /home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/accelerate/accelerator.py:432: FutureWarning: Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches', 'split_batches', 'even_batches', 'use_seedable_sampler']). Please pass an `accelerate.DataLoaderConfiguration` instead: 
  dataloader_config = DataLoaderConfiguration(dispatch_batches=None, split_batches=False, even_batches=True, use_seedable_sampler=True)
    warnings.warn(
  [INFO|trainer.py:607] 2024-03-27 21:46:52,604 >> Using auto half precision backend
  [INFO|configuration_utils.py:726] 2024-03-27 21:46:52,652 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/config.json
  [INFO|configuration_utils.py:789] 2024-03-27 21:46:52,653 >> Model config LlamaConfig {
    "_name_or_path": "meta-llama/Llama-2-7b-hf",
    "architectures": [
      "LlamaForCausalLM"
    ],
    "attention_bias": false,
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 4096,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 32,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "rope_theta": 10000.0,
    "tie_word_embeddings": false,
    "torch_dtype": "float16",
    "transformers_version": "4.39.1",
    "use_cache": true,
    "vocab_size": 32000
  }

  [INFO|modeling_utils.py:3283] 2024-03-27 21:46:52,654 >> loading weights file model.safetensors from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/model.safetensors.index.json
  [INFO|modeling_utils.py:1417] 2024-03-27 21:46:52,655 >> Instantiating LlamaForCausalLM model under default dtype torch.float16.
  [INFO|configuration_utils.py:928] 2024-03-27 21:46:52,656 >> Generate config GenerationConfig {
    "bos_token_id": 1,
    "eos_token_id": 2
  }

  Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.11it/s]
  [INFO|modeling_utils.py:4024] 2024-03-27 21:46:54,677 >> All model checkpoint weights were used when initializing LlamaForCausalLM.

  [INFO|modeling_utils.py:4032] 2024-03-27 21:46:54,677 >> All the weights of LlamaForCausalLM were initialized from the model checkpoint at meta-llama/Llama-2-7b-hf.
  If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
  [INFO|configuration_utils.py:883] 2024-03-27 21:46:54,734 >> loading configuration file generation_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/generation_config.json
  [INFO|configuration_utils.py:928] 2024-03-27 21:46:54,734 >> Generate config GenerationConfig {
    "bos_token_id": 1,
    "do_sample": true,
    "eos_token_id": 2,
    "max_length": 4096,
    "pad_token_id": 0,
    "temperature": 0.6,
    "top_p": 0.9
  }

  03/27/2024 21:46:54 - WARNING - root - Some parameters are on the meta device device because they were offloaded to the cpu.
  [INFO|trainer.py:1969] 2024-03-27 21:46:54,999 >> ***** Running training *****
  [INFO|trainer.py:1970] 2024-03-27 21:46:54,999 >>   Num examples = 5,285
  [INFO|trainer.py:1971] 2024-03-27 21:46:54,999 >>   Num Epochs = 1
  [INFO|trainer.py:1972] 2024-03-27 21:46:54,999 >>   Instantaneous batch size per device = 1
  [INFO|trainer.py:1975] 2024-03-27 21:46:54,999 >>   Total train batch size (w. parallel, distributed & accumulation) = 128
  [INFO|trainer.py:1976] 2024-03-27 21:46:54,999 >>   Gradient Accumulation steps = 128
  [INFO|trainer.py:1977] 2024-03-27 21:46:54,999 >>   Total optimization steps = 41
  [INFO|trainer.py:1978] 2024-03-27 21:46:55,003 >>   Number of trainable parameters = 450,887,680
  [INFO|integration_utils.py:723] 2024-03-27 21:46:55,006 >> Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"
  wandb: Currently logged in as: vashistt (cmu-anlp-project). Use `wandb login --relogin` to force relogin
  wandb: Tracking run with wandb version 0.16.5
  wandb: Run data is saved locally in /home/vashistt/anlp-project/lora_ft/wandb/run-20240327_214655-akw9sye5
  wandb: Run `wandb offline` to turn off syncing.
  wandb: Syncing run misty-oath-5
  wandb: ⭐️ View project at https://wandb.ai/cmu-anlp-project/huggingface
  wandb: 🚀 View run at https://wandb.ai/cmu-anlp-project/huggingface/runs/akw9sye5/workspace
    0%|                                                                                                                                                                 | 0/41 [00:00<?, ?it/s]Traceback (most recent call last):
    File "/home/vashistt/anlp-project/lora_ft/finetune_lm.py", line 852, in <module>
      main()
    File "/home/vashistt/anlp-project/lora_ft/finetune_lm.py", line 782, in main
      train_result = trainer.train(resume_from_checkpoint=checkpoint)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/transformers/trainer.py", line 1780, in train
      return inner_training_loop(
            ^^^^^^^^^^^^^^^^^^^^
    File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/transformers/trainer.py", line 2118, in _inner_training_loop
      tr_loss_step = self.training_step(model, inputs)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/transformers/trainer.py", line 3045, in training_step
      self.accelerator.backward(loss)
    File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/accelerate/accelerator.py", line 2001, in backward
      loss.backward(**kwargs)
    File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/torch/_tensor.py", line 522, in backward
      torch.autograd.backward(
    File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/torch/autograd/__init__.py", line 266, in backward
      Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
  RuntimeError: Found dtype Half but expected Float
  wandb: 🚀 View run misty-oath-5 at: https://wandb.ai/cmu-anlp-project/huggingface/runs/akw9sye5/workspace
  '''