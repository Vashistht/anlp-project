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



# Eval run (ran till the pruned model)

(prune_llm) [vashistt@lovelace lora_ft]$ CUDA_VISIBLE_DEVICES=8 python3 Run_evals.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512        --lora_r 128    --learning_rate 1e-4            --lora_alpha_ratio 4    --per_device_train_batch_size 1         --per_device_eval_batch_size 8       --do_train      --do_eval       --max_train_samples 15000       --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"     --hidden_mse_weight 0.0         --kl_weight 0.01        --dataset_name "wikitext"
03/28/2024 17:16:37 - WARNING - __main__ - Process rank: 0, device: cuda:0, n_gpu: 1distributed training: True, 16-bits training: False
03/28/2024 17:16:37 - INFO - __main__ - Training/evaluation parameters TrainingArguments(
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
logging_dir=/home/vashistt/anlp-project/finetuned_model/runs/Mar28_17-16-37_lovelace.ece.local.cmu.edu,
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
03/28/2024 17:16:42 - INFO - datasets.builder - Using custom data configuration default-a3e66ef7800043cd
Loading Dataset Infos from /home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/datasets/packaged_modules/json
03/28/2024 17:16:42 - INFO - datasets.info - Loading Dataset Infos from /home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/datasets/packaged_modules/json
Overwrite dataset info from restored data version if exists.
03/28/2024 17:16:42 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
03/28/2024 17:16:42 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
Found cached dataset c4 (/home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2)
03/28/2024 17:16:42 - INFO - datasets.builder - Found cached dataset c4 (/home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2)
Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
03/28/2024 17:16:42 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/datasets/load.py:2516: FutureWarning: 'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.
You can remove this warning by passing 'token=<use_auth_token>' instead.
  warnings.warn(
Overwrite dataset info from restored data version if exists.
03/28/2024 17:16:44 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
03/28/2024 17:16:44 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
03/28/2024 17:16:44 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
03/28/2024 17:16:44 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
Overwrite dataset info from restored data version if exists.
03/28/2024 17:16:45 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
03/28/2024 17:16:45 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
03/28/2024 17:16:45 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
03/28/2024 17:16:45 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
[INFO|configuration_utils.py:726] 2024-03-28 17:16:46,002 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/config.json
[INFO|configuration_utils.py:789] 2024-03-28 17:16:46,003 >> Model config LlamaConfig {
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

[INFO|tokenization_utils_base.py:2084] 2024-03-28 17:16:46,040 >> loading file tokenizer.model from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer.model
[INFO|tokenization_utils_base.py:2084] 2024-03-28 17:16:46,040 >> loading file added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:2084] 2024-03-28 17:16:46,040 >> loading file special_tokens_map.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/special_tokens_map.json
[INFO|tokenization_utils_base.py:2084] 2024-03-28 17:16:46,040 >> loading file tokenizer_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer_config.json
[INFO|tokenization_utils_base.py:2084] 2024-03-28 17:16:46,040 >> loading file tokenizer.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer.json
[INFO|configuration_utils.py:726] 2024-03-28 17:16:46,169 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/config.json
[INFO|configuration_utils.py:789] 2024-03-28 17:16:46,170 >> Model config LlamaConfig {
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

[INFO|modeling_utils.py:3283] 2024-03-28 17:16:46,173 >> loading weights file model.safetensors from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/model.safetensors.index.json
[INFO|modeling_utils.py:1417] 2024-03-28 17:16:46,174 >> Instantiating LlamaForCausalLM model under default dtype torch.float16.
[INFO|configuration_utils.py:928] 2024-03-28 17:16:46,175 >> Generate config GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2
}

Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.80s/it]
[INFO|modeling_utils.py:4024] 2024-03-28 17:16:52,586 >> All model checkpoint weights were used when initializing LlamaForCausalLM.

[INFO|modeling_utils.py:4032] 2024-03-28 17:16:52,586 >> All the weights of LlamaForCausalLM were initialized from the model checkpoint at meta-llama/Llama-2-7b-hf.
If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
[INFO|configuration_utils.py:883] 2024-03-28 17:16:52,626 >> loading configuration file generation_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/generation_config.json
[INFO|configuration_utils.py:928] 2024-03-28 17:16:52,626 >> Generate config GenerationConfig {
  "bos_token_id": 1,
  "do_sample": true,
  "eos_token_id": 2,
  "max_length": 4096,
  "pad_token_id": 0,
  "temperature": 0.6,
  "top_p": 0.9
}

Num params = :  6476271616
03/28/2024 17:16:52 - INFO - __main__ - *** Evaluate ***
Overwrite dataset info from restored data version if exists.
03/28/2024 17:16:54 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
03/28/2024 17:16:54 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
03/28/2024 17:16:54 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
03/28/2024 17:16:54 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
 99%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌ | 83/84 [00:47<00:00,  1.76it/s]
Original perplexity on wikitext = 5.110
Num params = :  6476271616
eleuther eval for original model
[INFO|configuration_utils.py:726] 2024-03-28 17:17:45,122 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/config.json
[INFO|configuration_utils.py:789] 2024-03-28 17:17:45,123 >> Model config LlamaConfig {
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

[INFO|tokenization_utils_base.py:2084] 2024-03-28 17:17:45,165 >> loading file tokenizer.model from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer.model
[INFO|tokenization_utils_base.py:2084] 2024-03-28 17:17:45,165 >> loading file tokenizer.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer.json
[INFO|tokenization_utils_base.py:2084] 2024-03-28 17:17:45,165 >> loading file added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:2084] 2024-03-28 17:17:45,165 >> loading file special_tokens_map.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/special_tokens_map.json
[INFO|tokenization_utils_base.py:2084] 2024-03-28 17:17:45,165 >> loading file tokenizer_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer_config.json
[INFO|configuration_utils.py:726] 2024-03-28 17:17:45,282 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/config.json
[INFO|configuration_utils.py:789] 2024-03-28 17:17:45,282 >> Model config LlamaConfig {
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

[INFO|modeling_utils.py:3283] 2024-03-28 17:17:45,283 >> loading weights file model.safetensors from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/model.safetensors.index.json
[INFO|modeling_utils.py:1417] 2024-03-28 17:17:45,284 >> Instantiating LlamaForCausalLM model under default dtype torch.float16.
[INFO|configuration_utils.py:928] 2024-03-28 17:17:45,284 >> Generate config GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2
}

Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.35it/s]
[INFO|modeling_utils.py:4024] 2024-03-28 17:17:46,894 >> All model checkpoint weights were used when initializing LlamaForCausalLM.

[INFO|modeling_utils.py:4032] 2024-03-28 17:17:46,894 >> All the weights of LlamaForCausalLM were initialized from the model checkpoint at meta-llama/Llama-2-7b-hf.
If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
[INFO|configuration_utils.py:883] 2024-03-28 17:17:46,932 >> loading configuration file generation_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/generation_config.json
[INFO|configuration_utils.py:928] 2024-03-28 17:17:46,933 >> Generate config GenerationConfig {
  "bos_token_id": 1,
  "do_sample": true,
  "eos_token_id": 2,
  "max_length": 4096,
  "pad_token_id": 0,
  "temperature": 0.6,
  "top_p": 0.9
}

We have loaded the new model !
Overwrite dataset info from restored data version if exists.
03/28/2024 17:17:50 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/gsm8k/main/0.0.0/e53f048856ff4f594e959d75785d2c2d37b678ee
03/28/2024 17:17:50 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/gsm8k/main/0.0.0/e53f048856ff4f594e959d75785d2c2d37b678ee
Found cached dataset gsm8k (/home/vashistt/.cache/huggingface/datasets/gsm8k/main/0.0.0/e53f048856ff4f594e959d75785d2c2d37b678ee)
03/28/2024 17:17:50 - INFO - datasets.builder - Found cached dataset gsm8k (/home/vashistt/.cache/huggingface/datasets/gsm8k/main/0.0.0/e53f048856ff4f594e959d75785d2c2d37b678ee)
Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/gsm8k/main/0.0.0/e53f048856ff4f594e959d75785d2c2d37b678ee
03/28/2024 17:17:50 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/gsm8k/main/0.0.0/e53f048856ff4f594e959d75785d2c2d37b678ee
Task: gsm8k; number of docs: 1319
Task: gsm8k; document 0; context prompt (starting on next line):
Question: Mr. Bodhi is transporting some animals using a yacht across a river. He has 20 cows, 15 foxes and three times as many zebras as foxes. To balance the yacht to ensure a smooth sail across the river, the total number of animals in the yacht needs to be 100. If he decides to add sheep to the yacht to make the yacht sail-worthy, how many sheep did he add to the yacht?
Answer: The number of cows and foxes in the yacht is 20+15 = <<20+15=35>>35
Mr. Bodhi also has three times as many zebras as foxes in the yacht, equal to 3*15 = <<3*15=45>>45 zebras.
The number of animals in the yacht so far is 35+45 = <<35+45=80>>80
To balance the yacht, Mr. Bodhi needs to add 100-80= <<100-80=20>>20 sheep
#### 20

Question: Manny is making lasagna for dinner with his four friends, Lisa, Raphael, Aaron, and Kai. He needs to know how many pieces to cut the lasagna into to serve it. Manny only wants one piece. Aaron doesn't like lasagna much and will probably only eat garlic bread and salad. Kai is always hungry and will eat twice as much as Manny. Raphael always eats half the amount Manny does, but his sister Lisa loves lasagna and will eat two pieces, plus any Raphael has left of his piece. How many pieces should Manny cut his lasagna into?
Answer: Manny will eat 1 piece.
Aaron will eat 0 pieces.
Kai will eat twice as much as Manny, so he will eat 2 * 1 = <<2*1=2>>2 pieces.
Raphael will eat half as much as Manny, so he will eat 1 * 1/2 = 1/2 piece.
Lisa will eat 2 pieces plus the remainder of Raphael’s piece, so she will eat 2 + 1/2 = 2 1/2 pieces.
Together, they will eat 1 + 0 + 2 + 1/2 + 2 1/2 = 1 + 2 + 3 = 6 pieces.
Thus, Manny should cut his lasagna into 6 pieces.
#### 6

Question: Barbara asked the butcher for 4 1/2 pound steaks that cost $15.00/pound.  She also asked for a pound and half of chicken breasts that were $8.00 a pound.  How much did she spend at the butchers?
Answer: She ordered 4 1/2 pound steaks so that's 4*.5 = <<4*.5=2>>2 pounds of steak.
The steak cost $15.00 a pound and she bought 2 pounds so that's 15*2 = $<<15*2=30.00>>30.00 for 4 steaks.
She also needed 1.5 pounds of chicken breasts at $8.00 a pound so that's 1.5*8 = $<<1.5*8=12.00>>12.00 for chicken.
The steaks cost $30.00 and the chicken cost $12.00 for a total of 30+12 = $<<30+12=42.00>>42.00 spent at the butchers.
#### 42

Question: There are 400 students. 120 students take dance as their elective. 200 students take art as their elective. The rest take music. What percentage of students take music?
Answer: There are 400-120-200=<<400-120-200=80>>80 students in music.
Thus, students in music make up (80/400)*100=<<80/400*100=20>>20% of the students.
#### 20

Question: John starts at an elevation of 400 feet.  He travels downward at a rate of 10 feet down per minute for 5 minutes.  What is his elevation now?
Answer: He traveled down 10*5=<<10*5=50>>50 feet.
So he is at an elevation of 400-50=<<400-50=350>>350 feet.
#### 350

Question: Jared is trying to increase his typing speed. He starts with 47 words per minute (WPM). After some lessons the next time he tests his typing speed it has increased to 52 WPM. If he continues to increase his typing speed once more by 5 words, what will be the average of the three measurements?
Answer:
(end of prompt on previous line)
Requests: Req_greedy_until("Question: Mr. Bodhi is transporting some animals using a yacht across a river. He has 20 cows, 15 foxes and three times as many zebras as foxes. To balance the yacht to ensure a smooth sail across the river, the total number of animals in the yacht needs to be 100. If he decides to add sheep to the yacht to make the yacht sail-worthy, how many sheep did he add to the yacht?\nAnswer: The number of cows and foxes in the yacht is 20+15 = <<20+15=35>>35\nMr. Bodhi also has three times as many zebras as foxes in the yacht, equal to 3*15 = <<3*15=45>>45 zebras.\nThe number of animals in the yacht so far is 35+45 = <<35+45=80>>80\nTo balance the yacht, Mr. Bodhi needs to add 100-80= <<100-80=20>>20 sheep\n#### 20\n\nQuestion: Manny is making lasagna for dinner with his four friends, Lisa, Raphael, Aaron, and Kai. He needs to know how many pieces to cut the lasagna into to serve it. Manny only wants one piece. Aaron doesn't like lasagna much and will probably only eat garlic bread and salad. Kai is always hungry and will eat twice as much as Manny. Raphael always eats half the amount Manny does, but his sister Lisa loves lasagna and will eat two pieces, plus any Raphael has left of his piece. How many pieces should Manny cut his lasagna into?\nAnswer: Manny will eat 1 piece.\nAaron will eat 0 pieces.\nKai will eat twice as much as Manny, so he will eat 2 * 1 = <<2*1=2>>2 pieces.\nRaphael will eat half as much as Manny, so he will eat 1 * 1/2 = 1/2 piece.\nLisa will eat 2 pieces plus the remainder of Raphael’s piece, so she will eat 2 + 1/2 = 2 1/2 pieces.\nTogether, they will eat 1 + 0 + 2 + 1/2 + 2 1/2 = 1 + 2 + 3 = 6 pieces.\nThus, Manny should cut his lasagna into 6 pieces.\n#### 6\n\nQuestion: Barbara asked the butcher for 4 1/2 pound steaks that cost $15.00/pound.  She also asked for a pound and half of chicken breasts that were $8.00 a pound.  How much did she spend at the butchers?\nAnswer: She ordered 4 1/2 pound steaks so that's 4*.5 = <<4*.5=2>>2 pounds of steak.\nThe steak cost $15.00 a pound and she bought 2 pounds so that's 15*2 = $<<15*2=30.00>>30.00 for 4 steaks.\nShe also needed 1.5 pounds of chicken breasts at $8.00 a pound so that's 1.5*8 = $<<1.5*8=12.00>>12.00 for chicken.\nThe steaks cost $30.00 and the chicken cost $12.00 for a total of 30+12 = $<<30+12=42.00>>42.00 spent at the butchers.\n#### 42\n\nQuestion: There are 400 students. 120 students take dance as their elective. 200 students take art as their elective. The rest take music. What percentage of students take music?\nAnswer: There are 400-120-200=<<400-120-200=80>>80 students in music.\nThus, students in music make up (80/400)*100=<<80/400*100=20>>20% of the students.\n#### 20\n\nQuestion: John starts at an elevation of 400 feet.  He travels downward at a rate of 10 feet down per minute for 5 minutes.  What is his elevation now?\nAnswer: He traveled down 10*5=<<10*5=50>>50 feet.\nSo he is at an elevation of 400-50=<<400-50=350>>350 feet.\n#### 350\n\nQuestion: Jared is trying to increase his typing speed. He starts with 47 words per minute (WPM). After some lessons the next time he tests his typing speed it has increased to 52 WPM. If he continues to increase his typing speed once more by 5 words, what will be the average of the three measurements?\nAnswer:", {'until': [':', 'Question:', 'Question']})[None]

Running greedy_until requests
  0%|                                                                                                                                   | 0/1319 [00:00<?, ?it/s]/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/transformers/generation/configuration_utils.py:492: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0.6` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
  warnings.warn(
/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/transformers/generation/configuration_utils.py:497: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
  warnings.warn(
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [1:28:06<00:00,  4.01s/it]
{'results': {'gsm8k': {'acc': 0.14859742228961334, 'acc_stderr': 0.009797503180527934}}}
original model param count : 6476271616
epoch 1, param count is 5182283776
epoch 2, param count is 3887034368
epoch 3, param count is 3239133184
Final model sparsity is : 0.500 
Final model param count : 3239133184
Num params = :  3239133184
03/28/2024 18:47:40 - INFO - __main__ - *** Evaluate ***
Overwrite dataset info from restored data version if exists.
03/28/2024 18:47:41 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
03/28/2024 18:47:41 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
03/28/2024 18:47:41 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
03/28/2024 18:47:41 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  0%|                                                                                                                                     | 0/84 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/vashistt/anlp-project/lora_ft/Run_evals.py", line 893, in <module>
    main()
  File "/home/vashistt/anlp-project/lora_ft/Run_evals.py", line 694, in main
    before_train_ppl, final_runtime = evaluate_ppl(data_args.dataset_name, model, tokenizer, model.seqlen)
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vashistt/anlp-project/lora_ft/evaluate_ppl.py", line 62, in evaluate_ppl
    outputs = model(input_ids, labels=target_ids)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py", line 1196, in forward
    outputs = self.model(
              ^^^^^^^^^^^
  File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py", line 1016, in forward
    layer_outputs = decoder_layer(
                    ^^^^^^^^^^^^^^
  File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py", line 739, in forward
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
                                                          ^^^^^^^^^^^^^^^
  File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/vashistt/ENTER/envs/prune_llm/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py", line 641, in forward
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: shape '[1, 4096, 32, 128]' is invalid for input of size 7864320

'''