# 1. llama2


## 1.1 - wikitext ppl
```
    (prune_llm_2) [vashistt@lovelace lora_ft]$ CUDA_VISIBLE_DEVICES=0 python3 Run_evals.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512        --lora_r 128    --learning_rate 1e-4            --lora_alpha_ratio 4    --per_device_train_batch_size 1         --per_device_eval_batch_size 8       --do_train      --do_eval       --max_train_samples 15000       --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"     --hidden_mse_weight 0.0         --kl_weight 0.01        --dataset_name "wikitext"
    03/28/2024 22:16:56 - WARNING - __main__ - Process rank: -1, device: cuda:0, n_gpu: 1distributed training: False, 16-bits training: False
    03/28/2024 22:16:56 - INFO - __main__ - Training/evaluation parameters TrainingArguments(
    _n_gpu=1,
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
    dataloader_pin_memory=True,
    ddp_bucket_cap_mb=None,
    ddp_find_unused_parameters=None,
    ddp_timeout=1800,
    debug=[],
    deepspeed=None,
    disable_tqdm=False,
    do_eval=True,
    do_predict=False,
    do_train=True,
    eval_accumulation_steps=None,
    eval_delay=0,
    eval_steps=None,
    evaluation_strategy=no,
    fp16=False,
    fp16_backend=auto,
    fp16_full_eval=False,
    fp16_opt_level=O1,
    fsdp=[],
    fsdp_config={'fsdp_min_num_params': 0, 'xla': False, 'xla_fsdp_grad_ckpt': False},
    fsdp_min_num_params=0,
    fsdp_transformer_layer_cls_to_wrap=None,
    full_determinism=False,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    greater_is_better=None,
    group_by_length=False,
    half_precision_backend=auto,
    hub_model_id=None,
    hub_private_repo=False,
    hub_strategy=every_save,
    hub_token=<HUB_TOKEN>,
    ignore_data_skip=False,
    include_inputs_for_metrics=False,
    jit_mode_eval=False,
    label_names=None,
    label_smoothing_factor=0.0,
    learning_rate=0.0001,
    length_column_name=length,
    load_best_model_at_end=False,
    local_rank=-1,
    log_level=passive,
    log_level_replica=warning,
    log_on_each_node=True,
    logging_dir=/home/vashistt/anlp-project/finetuned_model/runs/Mar28_22-16-56_lovelace.ece.local.cmu.edu,
    logging_first_step=False,
    logging_nan_inf_filter=True,
    logging_steps=500,
    logging_strategy=steps,
    lr_scheduler_type=linear,
    max_grad_norm=1.0,
    max_steps=-1,
    metric_for_best_model=None,
    mp_parameters=,
    no_cuda=False,
    num_train_epochs=1.0,
    optim=adamw_hf,
    optim_args=None,
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
    save_safetensors=False,
    save_steps=500,
    save_strategy=steps,
    save_total_limit=None,
    seed=42,
    sharded_ddp=[],
    skip_memory_metrics=True,
    tf32=None,
    torch_compile=False,
    torch_compile_backend=None,
    torch_compile_mode=None,
    torchdynamo=None,
    tpu_metrics_debug=False,
    tpu_num_cores=None,
    use_ipex=False,
    use_legacy_prediction_loop=False,
    use_mps_device=False,
    warmup_ratio=0.0,
    warmup_steps=0,
    weight_decay=0.0,
    xpu_backend=None,
    )
    Using custom data configuration default-a3e66ef7800043cd
    03/28/2024 22:17:01 - INFO - datasets.builder - Using custom data configuration default-a3e66ef7800043cd
    Loading Dataset Infos from /home/vashistt/ENTER/envs/prune_llm_2/lib/python3.9/site-packages/datasets/packaged_modules/json
    03/28/2024 22:17:01 - INFO - datasets.info - Loading Dataset Infos from /home/vashistt/ENTER/envs/prune_llm_2/lib/python3.9/site-packages/datasets/packaged_modules/json
    Overwrite dataset info from restored data version if exists.
    03/28/2024 22:17:01 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
    Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
    03/28/2024 22:17:01 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
    Found cached dataset c4 (/home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2)
    03/28/2024 22:17:01 - INFO - datasets.builder - Found cached dataset c4 (/home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2)
    Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
    03/28/2024 22:17:01 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
    /home/vashistt/ENTER/envs/prune_llm_2/lib/python3.9/site-packages/datasets/load.py:2516: FutureWarning: 'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.
    You can remove this warning by passing 'token=<use_auth_token>' instead.
      warnings.warn(
    Overwrite dataset info from restored data version if exists.
    03/28/2024 22:17:03 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
    Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
    03/28/2024 22:17:03 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
    Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
    03/28/2024 22:17:03 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
    Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
    03/28/2024 22:17:03 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
    Overwrite dataset info from restored data version if exists.
    03/28/2024 22:17:05 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
    Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
    03/28/2024 22:17:05 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
    Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
    03/28/2024 22:17:05 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
    Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
    03/28/2024 22:17:05 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
    [INFO|configuration_utils.py:668] 2024-03-28 22:17:05,827 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/config.json
    [INFO|configuration_utils.py:720] 2024-03-28 22:17:05,828 >> Model config LlamaConfig {
      "_name_or_path": "meta-llama/Llama-2-7b-hf",
      "architectures": [
        "LlamaForCausalLM"
      ],
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
      "pad_token_id": 0,
      "pretraining_tp": 1,
      "rms_norm_eps": 1e-05,
      "rope_scaling": null,
      "tie_word_embeddings": false,
      "torch_dtype": "float16",
      "transformers_version": "4.28.0",
      "use_cache": true,
      "vocab_size": 32000
    }

    [INFO|tokenization_utils_base.py:1809] 2024-03-28 22:17:05,866 >> loading file tokenizer.model from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer.model
    [INFO|tokenization_utils_base.py:1809] 2024-03-28 22:17:05,867 >> loading file added_tokens.json from cache at None
    [INFO|tokenization_utils_base.py:1809] 2024-03-28 22:17:05,867 >> loading file special_tokens_map.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/special_tokens_map.json
    [INFO|tokenization_utils_base.py:1809] 2024-03-28 22:17:05,867 >> loading file tokenizer_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer_config.json
    [INFO|configuration_utils.py:668] 2024-03-28 22:17:05,920 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/config.json
    [INFO|configuration_utils.py:720] 2024-03-28 22:17:05,921 >> Model config LlamaConfig {
      "_name_or_path": "meta-llama/Llama-2-7b-hf",
      "architectures": [
        "LlamaForCausalLM"
      ],
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
      "pad_token_id": 0,
      "pretraining_tp": 1,
      "rms_norm_eps": 1e-05,
      "rope_scaling": null,
      "tie_word_embeddings": false,
      "torch_dtype": "float16",
      "transformers_version": "4.28.0",
      "use_cache": true,
      "vocab_size": 32000
    }

    [INFO|modeling_utils.py:2534] 2024-03-28 22:17:05,924 >> loading weights file model.safetensors from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/model.safetensors.index.json
    [INFO|modeling_utils.py:1176] 2024-03-28 22:17:05,925 >> Instantiating LlamaForCausalLM model under default dtype torch.float16.
    [INFO|configuration_utils.py:575] 2024-03-28 22:17:05,926 >> Generate config GenerationConfig {
      "_from_model_config": true,
      "bos_token_id": 1,
      "eos_token_id": 2,
      "pad_token_id": 0,
      "transformers_version": "4.28.0"
    }

    Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.34s/it]
    [INFO|modeling_utils.py:3190] 2024-03-28 22:17:09,226 >> All model checkpoint weights were used when initializing LlamaForCausalLM.

    [INFO|modeling_utils.py:3198] 2024-03-28 22:17:09,227 >> All the weights of LlamaForCausalLM were initialized from the model checkpoint at meta-llama/Llama-2-7b-hf.
    If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
    [INFO|configuration_utils.py:537] 2024-03-28 22:17:09,270 >> loading configuration file generation_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/generation_config.json
    [INFO|modeling_utils.py:2839] 2024-03-28 22:17:09,270 >> Generation config file not found, using a generation config created from the model config.
    Num params = :  6476271616
    03/28/2024 22:17:09 - INFO - __main__ - *** Evaluate ***
    Overwrite dataset info from restored data version if exists.
    03/28/2024 22:17:11 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
    Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
    03/28/2024 22:17:11 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
    Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
    03/28/2024 22:17:11 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
    Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
    03/28/2024 22:17:11 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
    99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▋ | 83/84 [01:43<00:01,  1.24s/it]
    Original perplexity on wikitext = 5.109
    Num params = :  6476271616
    eleuther eval for original model
    original model param count : 6476271616
    epoch 1, param count is 5182283776
    epoch 2, param count is 3887034368
    epoch 3, param count is 3239133184
    Final model sparsity is : 0.500 
    Final model param count : 3239133184
    Num params = :  3239133184
    03/28/2024 22:20:33 - INFO - __main__ - *** Evaluate ***
    Overwrite dataset info from restored data version if exists.
    03/28/2024 22:20:35 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
    Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
    03/28/2024 22:20:35 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
    Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
    03/28/2024 22:20:35 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
    Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
    03/28/2024 22:20:35 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
    99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▋ | 83/84 [00:53<00:00,  1.56it/s]
    [SpeedUp=2.159] Original perplexity on wikitext = 5.109 | Before Training perplexity on wikitext = 46.594
    [INFO|configuration_utils.py:668] 2024-03-28 22:21:30,859 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/config.json
    [INFO|configuration_utils.py:720] 2024-03-28 22:21:30,860 >> Model config LlamaConfig {
      "_name_or_path": "meta-llama/Llama-2-7b-hf",
      "architectures": [
        "LlamaForCausalLM"
      ],
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
      "pad_token_id": 0,
      "pretraining_tp": 1,
      "rms_norm_eps": 1e-05,
      "rope_scaling": null,
      "tie_word_embeddings": false,
      "torch_dtype": "float16",
      "transformers_version": "4.28.0",
      "use_cache": true,
      "vocab_size": 32000
    }

    [INFO|tokenization_utils_base.py:1809] 2024-03-28 22:21:30,903 >> loading file tokenizer.model from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer.model
    [INFO|tokenization_utils_base.py:1809] 2024-03-28 22:21:30,903 >> loading file tokenizer.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer.json
    [INFO|tokenization_utils_base.py:1809] 2024-03-28 22:21:30,903 >> loading file added_tokens.json from cache at None
    [INFO|tokenization_utils_base.py:1809] 2024-03-28 22:21:30,903 >> loading file special_tokens_map.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/special_tokens_map.json
    [INFO|tokenization_utils_base.py:1809] 2024-03-28 22:21:30,903 >> loading file tokenizer_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer_config.json
    [INFO|configuration_utils.py:668] 2024-03-28 22:21:30,991 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/config.json
    [INFO|configuration_utils.py:720] 2024-03-28 22:21:30,991 >> Model config LlamaConfig {
      "_name_or_path": "meta-llama/Llama-2-7b-hf",
      "architectures": [
        "LlamaForCausalLM"
      ],
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
      "pad_token_id": 0,
      "pretraining_tp": 1,
      "rms_norm_eps": 1e-05,
      "rope_scaling": null,
      "tie_word_embeddings": false,
      "torch_dtype": "float16",
      "transformers_version": "4.28.0",
      "use_cache": true,
      "vocab_size": 32000
    }

    [INFO|modeling_utils.py:2534] 2024-03-28 22:21:30,992 >> loading weights file model.safetensors from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/model.safetensors.index.json
    [INFO|modeling_utils.py:1176] 2024-03-28 22:21:30,993 >> Instantiating LlamaForCausalLM model under default dtype torch.float16.
    [INFO|configuration_utils.py:575] 2024-03-28 22:21:30,993 >> Generate config GenerationConfig {
      "_from_model_config": true,
      "bos_token_id": 1,
      "eos_token_id": 2,
      "pad_token_id": 0,
      "transformers_version": "4.28.0"
    }
'''

## 1.2 - boolq

```
  [IN████████████████████████████████████████████████████| 3.85M/3.85M [00:00<00:00, 11.7MB/s]
  storing hf://datasets/super_glue@9f1f8088b6705f471970f4f2edbcdb450ccf8b22/boolq/train/0000.parquet in cache at /home/vashistt/.cache/huggingface/datasets/downloads/c36693493a9c745729d14c9b060a6f232d51df20e507c84514df549c87685038
  03/28/2024 23:14:25 - INFO - datasets.utils.file_utils - storing hf://datasets/super_glue@9f1f8088b6705f471970f4f2edbcdb450ccf8b22/boolq/train/0000.parquet in cache at /home/vashistt/.cache/huggingface/datasets/downloads/c36693493a9c745729d14c9b060a6f232d51df20e507c84514df549c87685038
  creating metadata file for /home/vashistt/.cache/huggingface/datasets/downloads/c36693493a9c745729d14c9b060a6f232d51df20e507c84514df549c87685038
  03/28/2024 23:14:25 - INFO - datasets.utils.file_utils - creating metadata file for /home/vashistt/.cache/huggingface/datasets/downloads/c36693493a9c745729d14c9b060a6f232d51df20e507c84514df549c87685038
  hf://datasets/super_glue@9f1f8088b6705f471970f4f2edbcdb450ccf8b22/boolq/validation/0000.parquet not found in cache or force_download set to True, downloading to /home/vashistt/.cache/huggingface/datasets/downloads/5301eba99c8e94af568385df0e3bfa40ff85cad2353dbf14529f09c6e24999d6.incomplete
  03/28/2024 23:14:25 - INFO - datasets.utils.file_utils - hf://datasets/super_glue@9f1f8088b6705f471970f4f2edbcdb450ccf8b22/boolq/validation/0000.parquet not found in cache or force_download set to True, downloading to /home/vashistt/.cache/huggingface/datasets/downloads/5301eba99c8e94af568385df0e3bfa40ff85cad2353dbf14529f09c6e24999d6.incomplete
  Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1.31M/1.31M [00:00<00:00, 8.95MB/s]
  storing hf://datasets/super_glue@9f1f8088b6705f471970f4f2edbcdb450ccf8b22/boolq/validation/0000.parquet in cache at /home/vashistt/.cache/huggingface/datasets/downloads/5301eba99c8e94af568385df0e3bfa40ff85cad2353dbf14529f09c6e24999d6
  03/28/2024 23:14:26 - INFO - datasets.utils.file_utils - storing hf://datasets/super_glue@9f1f8088b6705f471970f4f2edbcdb450ccf8b22/boolq/validation/0000.parquet in cache at /home/vashistt/.cache/huggingface/datasets/downloads/5301eba99c8e94af568385df0e3bfa40ff85cad2353dbf14529f09c6e24999d6
  creating metadata file for /home/vashistt/.cache/huggingface/datasets/downloads/5301eba99c8e94af568385df0e3bfa40ff85cad2353dbf14529f09c6e24999d6
  03/28/2024 23:14:26 - INFO - datasets.utils.file_utils - creating metadata file for /home/vashistt/.cache/huggingface/datasets/downloads/5301eba99c8e94af568385df0e3bfa40ff85cad2353dbf14529f09c6e24999d6
  hf://datasets/super_glue@9f1f8088b6705f471970f4f2edbcdb450ccf8b22/boolq/test/0000.parquet not found in cache or force_download set to True, downloading to /home/vashistt/.cache/huggingface/datasets/downloads/755e326c75d588530e16aa25e248fa61adab2ac4fec54d27ad5be6c581d361e6.incomplete
      03/28/2024 23:14:26 - INFO - datasets.uFO|tokenization_utils_base.py:1809] 2024-03-28 23:11:44,130 >> loading file tokenizer.model from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer.model
      [INFO|tokenization_utils_base.py:1809] 2024-03-28 23:11:44,130 >> loading file tokenizer.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer.json
      [INFO|tokenization_utils_base.py:1809] 2024-03-28 23:11:44,130 >> loading file added_tokens.json from cache at None
      [INFO|tokenization_utils_base.py:1809] 2024-03-28 23:11:44,130 >> loading file special_tokens_map.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/special_tokens_map.json
      [INFO|tokenization_utils_base.py:1809] 2024-03-28 23:11:44,130 >> loading file tokenizer_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer_config.json
      [INFO|configuration_utils.py:668] 2024-03-28 23:11:44,223 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/config.json
      [INFO|configuration_utils.py:720] 2024-03-28 23:11:44,224 >> Model config LlamaConfig {
        "_name_or_path": "meta-llama/Llama-2-7b-hf",
        "architectures": [
          "LlamaForCausalLM"
        ],
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
        "pad_token_id": 0,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": null,
        "tie_word_embeddings": false,
        "torch_dtype": "float16",
        "transformers_version": "4.28.0",
        "use_cache": true,
        "vocab_size": 32000
      }

      [INFO|modeling_utils.py:2534] 2024-03-28 23:11:44,225 >> loading weights file model.safetensors from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/model.safetensors.index.json
      [INFO|modeling_utils.py:1176] 2024-03-28 23:11:44,225 >> Instantiating LlamaForCausalLM model under default dtype torch.float16.
      [INFO|configuration_utils.py:575] 2024-03-28 23:11:44,226 >> Generate config GenerationConfig {
        "_from_model_config": true,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "transformers_version": "4.28.0"
      }

      Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  2.10it/s]
      [INFO|modeling_utils.py:3190] 2024-03-28 23:14:21,890 >> All model checkpoint weights were used when initializing LlamaForCausalLM.

      [INFO|modeling_utils.py:3198] 2024-03-28 23:14:21,890 >> All the weights of LlamaForCausalLM were initialized from the model checkpoint at meta-llama/Llama-2-7b-hf.
      If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
      [INFO|configuration_utils.py:537] 2024-03-28 23:14:21,942 >> loading configuration file generation_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/generation_config.json
      [INFO|configuration_utils.py:575] 2024-03-28 23:14:21,943 >> Generate config GenerationConfig {
        "bos_token_id": 1,
        "do_sample": true,
        "eos_token_id": 2,
        "max_length": 4096,
        "pad_token_id": 0,
        "temperature": 0.6,
        "top_p": 0.9,
        "transformers_version": "4.28.0"
      }

      We have loaded the new model !
      Generating dataset super_glue (/home/vashistt/.cache/huggingface/datasets/super_glue/boolq/1.0.3/b051de3f07b5fd5ab80398a4836458db56234e24)
      03/28/2024 23:14:25 - INFO - datasets.builder - Generating dataset super_glue (/home/vashistt/.cache/huggingface/datasets/super_glue/boolq/1.0.3/b051de3f07b5fd5ab80398a4836458db56234e24)
      Downloading and preparing dataset super_glue/boolq (download: 3.93 MiB, generated: 9.91 MiB, post-processed: Unknown size, total: 13.84 MiB) to /home/vashistt/.cache/huggingface/datasets/super_glue/boolq/1.0.3/b051de3f07b5fd5ab80398a4836458db56234e24...
      03/28/2024 23:14:25 - INFO - datasets.builder - Downloading and preparing dataset super_glue/boolq (download: 3.93 MiB, generated: 9.91 MiB, post-processed: Unknown size, total: 13.84 MiB) to /home/vashistt/.cache/huggingface/datasets/super_glue/boolq/1.0.3/b051de3f07b5fd5ab80398a4836458db56234e24...
      Dataset not on Hf google storage. Downloading and preparing it from source
      03/28/2024 23:14:25 - INFO - datasets.builder - Dataset not on Hf google storage. Downloading and preparing it from source
      hf://datasets/super_glue@9f1f8088b6705f471970f4f2edbcdb450ccf8b22/boolq/train/0000.parquet not found in cache or force_download set to True, downloading to /home/vashistt/.cache/huggingface/datasets/downloads/c36693493a9c745729d14c9b060a6f232d51df20e507c84514df549c87685038.incomplete
      03/28/2024 23:14:25 - INFO - datasets.utils.file_utils - hf://datasets/super_glue@9f1f8088b6705f471970f4f2edbcdb450ccf8b22/boolq/train/0000.parquet not found in cache or force_download set to True, downloading to /home/vashistt/.cache/huggingface/datasets/downloads/c36693493a9c745729d14c9b060a6f232d51df20e507c84514df549c87685038.incomplete
      Downloading data: 100%|██████████████████████████████████████tils.file_utils - hf://datasets/super_glue@9f1f8088b6705f471970f4f2edbcdb450ccf8b22/boolq/test/0000.parquet not found in cache or force_download set to True, downloading to /home/vashistt/.cache/huggingface/datasets/downloads/755e326c75d588530e16aa25e248fa61adab2ac4fec54d27ad5be6c581d361e6.incomplete
      Downloading data: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1.31M/1.31M [00:00<00:00, 7.91MB/s]
      storing hf://datasets/super_glue@9f1f8088b6705f471970f4f2edbcdb450ccf8b22/boolq/test/0000.parquet in cache at /home/vashistt/.cache/huggingface/datasets/downloads/755e326c75d588530e16aa25e248fa61adab2ac4fec54d27ad5be6c581d361e6
      03/28/2024 23:14:26 - INFO - datasets.utils.file_utils - storing hf://datasets/super_glue@9f1f8088b6705f471970f4f2edbcdb450ccf8b22/boolq/test/0000.parquet in cache at /home/vashistt/.cache/huggingface/datasets/downloads/755e326c75d588530e16aa25e248fa61adab2ac4fec54d27ad5be6c581d361e6
      creating metadata file for /home/vashistt/.cache/huggingface/datasets/downloads/755e326c75d588530e16aa25e248fa61adab2ac4fec54d27ad5be6c581d361e6
      03/28/2024 23:14:26 - INFO - datasets.utils.file_utils - creating metadata file for /home/vashistt/.cache/huggingface/datasets/downloads/755e326c75d588530e16aa25e248fa61adab2ac4fec54d27ad5be6c581d361e6
      Downloading took 0.0 min
      03/28/2024 23:14:26 - INFO - datasets.download.download_manager - Downloading took 0.0 min
      Checksum Computation took 0.0 min
      03/28/2024 23:14:26 - INFO - datasets.download.download_manager - Checksum Computation took 0.0 min
      Generating train split
      03/28/2024 23:14:26 - INFO - datasets.builder - Generating train split
      Generating train split: 100%|██████████████████████████████████████████████████████████████████████████| 9427/9427 [00:00<00:00, 362693.01 examples/s]
      Generating validation split
      03/28/2024 23:14:26 - INFO - datasets.builder - Generating validation split
      Generating validation split: 100%|█████████████████████████████████████████████████████████████████████| 3270/3270 [00:00<00:00, 420691.19 examples/s]
      Generating test split
      03/28/2024 23:14:26 - INFO - datasets.builder - Generating test split
      Generating test split: 100%|███████████████████████████████████████████████████████████████████████████| 3245/3245 [00:00<00:00, 388339.32 examples/s]
      All the splits matched successfully.
      03/28/2024 23:14:26 - INFO - datasets.utils.info_utils - All the splits matched successfully.
      Dataset super_glue downloaded and prepared to /home/vashistt/.cache/huggingface/datasets/super_glue/boolq/1.0.3/b051de3f07b5fd5ab80398a4836458db56234e24. Subsequent calls will reuse this data.
      03/28/2024 23:14:26 - INFO - datasets.builder - Dataset super_glue downloaded and prepared to /home/vashistt/.cache/huggingface/datasets/super_glue/boolq/1.0.3/b051de3f07b5fd5ab80398a4836458db56234e24. Subsequent calls will reuse this data.
      Task: boolq; number of docs: 3270
      Task: boolq; document 0; context prompt (starting on next line):
      NCIS: New Orleans (season 4) -- The fourth season of NCIS: New Orleans premiered on September 26, 2017 on CBS. The series continues to air following Bull, Tuesday at 10:00 p.m. (ET) and contained 24 episodes. The season concluded on May 15, 2018.
      Question: is ncis new orleans over for the season?
      Answer:
      (end of prompt on previous line)
      Requests: (Req_loglikelihood('NCIS: New Orleans (season 4) -- The fourth season of NCIS: New Orleans premiered on September 26, 2017 on CBS. The series continues to air following Bull, Tuesday at 10:00 p.m. (ET) and contained 24 episodes. The season concluded on May 15, 2018.\nQuestion: is ncis new orleans over for the season?\nAnswer:', ' yes')[0]
      , Req_loglikelihood('NCIS: New Orleans (season 4) -- The fourth season of NCIS: New Orleans premiered on September 26, 2017 on CBS. The series continues to air following Bull, Tuesday at 10:00 p.m. (ET) and contained 24 episodes. The season concluded on May 15, 2018.\nQuestion: is ncis new orleans over for the season?\nAnswer:', ' no')[0]
      )
      Running loglikelihood requests
      100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6540/6540 [03:58<00:00, 27.46it/s]
      {'results': {'boolq': {'acc': 0.5174311926605505, 'acc_stderr': 0.008739739052380803}}}
'''

## 1.3  All datasets - wiki for llama 2 50%


```
  (prune_llm_2) [vashistt@lovelace lora_ft]$ location="/home/vashistt/anlp-project/outdir_llama_2_7b/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2-wikitext_Adaptive=Yes"
  (prune_llm_2) [vashistt@lovelace lora_ft]$ outdir="/home/vashistt/anlp-project/finetuned_model"
  (prune_llm_2) [vashistt@lovelace lora_ft]$ CUDA_VISIBLE_DEVICES=0 python3 Run_evals.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512        --lora_r 128    --learning_rate 1e-4            --lora_alpha_ratio 4    --per_device_train_batch_size 1         --per_device_eval_batch_size 8       --do_train      --do_eval       --max_train_samples 15000       --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"     --hidden_mse_weight 0.0         --kl_weight 0.01        --dataset_name "wikitext"
  03/29/2024 00:01:06 - WARNING - __main__ - Process rank: -1, device: cuda:0, n_gpu: 1distributed training: False, 16-bits training: False
  03/29/2024 00:01:06 - INFO - __main__ - Training/evaluation parameters TrainingArguments(
  _n_gpu=1,
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
  dataloader_pin_memory=True,
  ddp_bucket_cap_mb=None,
  ddp_find_unused_parameters=None,
  ddp_timeout=1800,
  debug=[],
  deepspeed=None,
  disable_tqdm=False,
  do_eval=True,
  do_predict=False,
  do_train=True,
  eval_accumulation_steps=None,
  eval_delay=0,
  eval_steps=None,
  evaluation_strategy=no,
  fp16=False,
  fp16_backend=auto,
  fp16_full_eval=False,
  fp16_opt_level=O1,
  fsdp=[],
  fsdp_config={'fsdp_min_num_params': 0, 'xla': False, 'xla_fsdp_grad_ckpt': False},
  fsdp_min_num_params=0,
  fsdp_transformer_layer_cls_to_wrap=None,
  full_determinism=False,
  gradient_accumulation_steps=1,
  gradient_checkpointing=False,
  greater_is_better=None,
  group_by_length=False,
  half_precision_backend=auto,
  hub_model_id=None,
  hub_private_repo=False,
  hub_strategy=every_save,
  hub_token=<HUB_TOKEN>,
  ignore_data_skip=False,
  include_inputs_for_metrics=False,
  jit_mode_eval=False,
  label_names=None,
  label_smoothing_factor=0.0,
  learning_rate=0.0001,
  length_column_name=length,
  load_best_model_at_end=False,
  local_rank=-1,
  log_level=passive,
  log_level_replica=warning,
  log_on_each_node=True,
  logging_dir=/home/vashistt/anlp-project/finetuned_model/runs/Mar29_00-01-06_lovelace.ece.local.cmu.edu,
  logging_first_step=False,
  logging_nan_inf_filter=True,
  logging_steps=500,
  logging_strategy=steps,
  lr_scheduler_type=linear,
  max_grad_norm=1.0,
  max_steps=-1,
  metric_for_best_model=None,
  mp_parameters=,
  no_cuda=False,
  num_train_epochs=1.0,
  optim=adamw_hf,
  optim_args=None,
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
  save_safetensors=False,
  save_steps=500,
  save_strategy=steps,
  save_total_limit=None,
  seed=42,
  sharded_ddp=[],
  skip_memory_metrics=True,
  tf32=None,
  torch_compile=False,
  torch_compile_backend=None,
  torch_compile_mode=None,
  torchdynamo=None,
  tpu_metrics_debug=False,
  tpu_num_cores=None,
  use_ipex=False,
  use_legacy_prediction_loop=False,
  use_mps_device=False,
  warmup_ratio=0.0,
  warmup_steps=0,
  weight_decay=0.0,
  xpu_backend=None,
  )
  Using custom data configuration default-a3e66ef7800043cd
  03/29/2024 00:01:11 - INFO - datasets.builder - Using custom data configuration default-a3e66ef7800043cd
  Loading Dataset Infos from /home/vashistt/ENTER/envs/prune_llm_2/lib/python3.9/site-packages/datasets/packaged_modules/json
  03/29/2024 00:01:11 - INFO - datasets.info - Loading Dataset Infos from /home/vashistt/ENTER/envs/prune_llm_2/lib/python3.9/site-packages/datasets/packaged_modules/json
  Overwrite dataset info from restored data version if exists.
  03/29/2024 00:01:11 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
  03/29/2024 00:01:11 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
  Found cached dataset c4 (/home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2)
  03/29/2024 00:01:11 - INFO - datasets.builder - Found cached dataset c4 (/home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
  03/29/2024 00:01:11 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
  /home/vashistt/ENTER/envs/prune_llm_2/lib/python3.9/site-packages/datasets/load.py:2516: FutureWarning: 'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.
  You can remove this warning by passing 'token=<use_auth_token>' instead.
    warnings.warn(
  Overwrite dataset info from restored data version if exists.
  03/29/2024 00:01:13 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/29/2024 00:01:13 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  03/29/2024 00:01:13 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/29/2024 00:01:13 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  Overwrite dataset info from restored data version if exists.
  03/29/2024 00:01:16 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/29/2024 00:01:16 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  03/29/2024 00:01:16 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/29/2024 00:01:16 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  [INFO|configuration_utils.py:668] 2024-03-29 00:01:16,444 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/config.json
  [INFO|configuration_utils.py:720] 2024-03-29 00:01:16,445 >> Model config LlamaConfig {
    "_name_or_path": "meta-llama/Llama-2-7b-hf",
    "architectures": [
      "LlamaForCausalLM"
    ],
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
    "pad_token_id": 0,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "tie_word_embeddings": false,
    "torch_dtype": "float16",
    "transformers_version": "4.28.0",
    "use_cache": true,
    "vocab_size": 32000
  }

  [INFO|tokenization_utils_base.py:1809] 2024-03-29 00:01:16,576 >> loading file tokenizer.model from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer.model
  [INFO|tokenization_utils_base.py:1809] 2024-03-29 00:01:16,576 >> loading file added_tokens.json from cache at None
  [INFO|tokenization_utils_base.py:1809] 2024-03-29 00:01:16,576 >> loading file special_tokens_map.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/special_tokens_map.json
  [INFO|tokenization_utils_base.py:1809] 2024-03-29 00:01:16,576 >> loading file tokenizer_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer_config.json
  [INFO|configuration_utils.py:668] 2024-03-29 00:01:16,618 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/config.json
  [INFO|configuration_utils.py:720] 2024-03-29 00:01:16,618 >> Model config LlamaConfig {
    "_name_or_path": "meta-llama/Llama-2-7b-hf",
    "architectures": [
      "LlamaForCausalLM"
    ],
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
    "pad_token_id": 0,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "tie_word_embeddings": false,
    "torch_dtype": "float16",
    "transformers_version": "4.28.0",
    "use_cache": true,
    "vocab_size": 32000
  }

  [INFO|modeling_utils.py:2534] 2024-03-29 00:01:16,621 >> loading weights file model.safetensors from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/model.safetensors.index.json
  [INFO|modeling_utils.py:1176] 2024-03-29 00:01:16,622 >> Instantiating LlamaForCausalLM model under default dtype torch.float16.
  [INFO|configuration_utils.py:575] 2024-03-29 00:01:16,622 >> Generate config GenerationConfig {
    "_from_model_config": true,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "transformers_version": "4.28.0"
  }

  Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.26s/it]
  [INFO|modeling_utils.py:3190] 2024-03-29 00:01:19,798 >> All model checkpoint weights were used when initializing LlamaForCausalLM.

  [INFO|modeling_utils.py:3198] 2024-03-29 00:01:19,798 >> All the weights of LlamaForCausalLM were initialized from the model checkpoint at meta-llama/Llama-2-7b-hf.
  If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
  [INFO|configuration_utils.py:537] 2024-03-29 00:01:19,866 >> loading configuration file generation_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/generation_config.json
  [INFO|modeling_utils.py:2839] 2024-03-29 00:01:19,866 >> Generation config file not found, using a generation config created from the model config.
  Num params = :  6476271616
  03/29/2024 00:01:19 - INFO - __main__ - *** Evaluate ***
  Num params = :  6476271616
  eleuther eval for original model
  original model param count : 6476271616
  epoch 1, param count is 5182283776
  epoch 2, param count is 3887034368
  epoch 3, param count is 3239133184
  Final model sparsity is : 0.500 
  Final model param count : 3239133184
  Num params = :  3239133184
  03/29/2024 00:03:00 - INFO - __main__ - *** Evaluate ***
  [INFO|configuration_utils.py:668] 2024-03-29 00:03:00,465 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/config.json
  [INFO|configuration_utils.py:720] 2024-03-29 00:03:00,466 >> Model config LlamaConfig {
    "_name_or_path": "meta-llama/Llama-2-7b-hf",
    "architectures": [
      "LlamaForCausalLM"
    ],
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
    "pad_token_id": 0,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "tie_word_embeddings": false,
    "torch_dtype": "float16",
    "transformers_version": "4.28.0",
    "use_cache": true,
    "vocab_size": 32000
  }

  [INFO|tokenization_utils_base.py:1809] 2024-03-29 00:03:00,504 >> loading file tokenizer.model from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer.model
  [INFO|tokenization_utils_base.py:1809] 2024-03-29 00:03:00,504 >> loading file tokenizer.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer.json
  [INFO|tokenization_utils_base.py:1809] 2024-03-29 00:03:00,504 >> loading file added_tokens.json from cache at None
  [INFO|tokenization_utils_base.py:1809] 2024-03-29 00:03:00,504 >> loading file special_tokens_map.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/special_tokens_map.json
  [INFO|tokenization_utils_base.py:1809] 2024-03-29 00:03:00,504 >> loading file tokenizer_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/tokenizer_config.json
  [INFO|configuration_utils.py:668] 2024-03-29 00:03:00,599 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/config.json
  [INFO|configuration_utils.py:720] 2024-03-29 00:03:00,599 >> Model config LlamaConfig {
    "_name_or_path": "meta-llama/Llama-2-7b-hf",
    "architectures": [
      "LlamaForCausalLM"
    ],
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
    "pad_token_id": 0,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "tie_word_embeddings": false,
    "torch_dtype": "float16",
    "transformers_version": "4.28.0",
    "use_cache": true,
    "vocab_size": 32000
  }

  [INFO|modeling_utils.py:2534] 2024-03-29 00:03:00,600 >> loading weights file model.safetensors from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/model.safetensors.index.json
  [INFO|modeling_utils.py:1176] 2024-03-29 00:03:00,601 >> Instantiating LlamaForCausalLM model under default dtype torch.float16.
  [INFO|configuration_utils.py:575] 2024-03-29 00:03:00,601 >> Generate config GenerationConfig {
    "_from_model_config": true,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "transformers_version": "4.28.0"
  }

  Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.92it/s]
  [INFO|modeling_utils.py:3190] 2024-03-29 00:05:39,345 >> All model checkpoint weights were used when initializing LlamaForCausalLM.

  [INFO|modeling_utils.py:3198] 2024-03-29 00:05:39,345 >> All the weights of LlamaForCausalLM were initialized from the model checkpoint at meta-llama/Llama-2-7b-hf.
  If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
  [INFO|configuration_utils.py:537] 2024-03-29 00:05:39,389 >> loading configuration file generation_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/8a0442e81540efaeb1a0fe3e95477b5e0edfd423/generation_config.json
  [INFO|configuration_utils.py:575] 2024-03-29 00:05:39,390 >> Generate config GenerationConfig {
    "bos_token_id": 1,
    "do_sample": true,
    "eos_token_id": 2,
    "max_length": 4096,
    "pad_token_id": 0,
    "temperature": 0.6,
    "top_p": 0.9,
    "transformers_version": "4.28.0"
  }

  We have loaded the new model !
  Overwrite dataset info from restored data version if exists.
  03/29/2024 00:05:42 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/winogrande/winogrande_xl/1.1.0/85ac5b5a3b7a930e22d590176e39460400d19e41
  03/29/2024 00:05:42 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/winogrande/winogrande_xl/1.1.0/85ac5b5a3b7a930e22d590176e39460400d19e41
  Found cached dataset winogrande (/home/vashistt/.cache/huggingface/datasets/winogrande/winogrande_xl/1.1.0/85ac5b5a3b7a930e22d590176e39460400d19e41)
  03/29/2024 00:05:42 - INFO - datasets.builder - Found cached dataset winogrande (/home/vashistt/.cache/huggingface/datasets/winogrande/winogrande_xl/1.1.0/85ac5b5a3b7a930e22d590176e39460400d19e41)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/winogrande/winogrande_xl/1.1.0/85ac5b5a3b7a930e22d590176e39460400d19e41
  03/29/2024 00:05:42 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/winogrande/winogrande_xl/1.1.0/85ac5b5a3b7a930e22d590176e39460400d19e41
  Overwrite dataset info from restored data version if exists.
  03/29/2024 00:05:43 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/super_glue/boolq/1.0.3/b051de3f07b5fd5ab80398a4836458db56234e24
  03/29/2024 00:05:43 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/super_glue/boolq/1.0.3/b051de3f07b5fd5ab80398a4836458db56234e24
  Found cached dataset super_glue (/home/vashistt/.cache/huggingface/datasets/super_glue/boolq/1.0.3/b051de3f07b5fd5ab80398a4836458db56234e24)
  03/29/2024 00:05:43 - INFO - datasets.builder - Found cached dataset super_glue (/home/vashistt/.cache/huggingface/datasets/super_glue/boolq/1.0.3/b051de3f07b5fd5ab80398a4836458db56234e24)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/super_glue/boolq/1.0.3/b051de3f07b5fd5ab80398a4836458db56234e24
  03/29/2024 00:05:43 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/super_glue/boolq/1.0.3/b051de3f07b5fd5ab80398a4836458db56234e24
  Overwrite dataset info from restored data version if exists.
  03/29/2024 00:05:47 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453
  03/29/2024 00:05:47 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453
  Found cached dataset ai2_arc (/home/vashistt/.cache/huggingface/datasets/ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453)
  03/29/2024 00:05:47 - INFO - datasets.builder - Found cached dataset ai2_arc (/home/vashistt/.cache/huggingface/datasets/ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453
  03/29/2024 00:05:47 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/ai2_arc/ARC-Challenge/0.0.0/210d026faf9955653af8916fad021475a3f00453
  Overwrite dataset info from restored data version if exists.
  03/29/2024 00:05:51 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/ai2_arc/ARC-Easy/0.0.0/210d026faf9955653af8916fad021475a3f00453
  03/29/2024 00:05:51 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/ai2_arc/ARC-Easy/0.0.0/210d026faf9955653af8916fad021475a3f00453
  Found cached dataset ai2_arc (/home/vashistt/.cache/huggingface/datasets/ai2_arc/ARC-Easy/0.0.0/210d026faf9955653af8916fad021475a3f00453)
  03/29/2024 00:05:51 - INFO - datasets.builder - Found cached dataset ai2_arc (/home/vashistt/.cache/huggingface/datasets/ai2_arc/ARC-Easy/0.0.0/210d026faf9955653af8916fad021475a3f00453)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/ai2_arc/ARC-Easy/0.0.0/210d026faf9955653af8916fad021475a3f00453
  03/29/2024 00:05:51 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/ai2_arc/ARC-Easy/0.0.0/210d026faf9955653af8916fad021475a3f00453
  /home/vashistt/ENTER/envs/prune_llm_2/lib/python3.9/site-packages/datasets/load.py:1461: FutureWarning: The repository for hellaswag contains custom code which must be executed to correctly load the dataset. You can inspect the repository content at https://hf.co/datasets/hellaswag
  You can avoid this message in future by passing the argument `trust_remote_code=True`.
  Passing `trust_remote_code=True` will be mandatory to load this dataset from the next major release of `datasets`.
    warnings.warn(
  Loading Dataset Infos from /home/vashistt/.cache/huggingface/modules/datasets_modules/datasets/hellaswag/512a66dd8b1b1643ab4a48aa4f150d04c91680da6a4096498a5e5f799623d5ae
  03/29/2024 00:05:52 - INFO - datasets.info - Loading Dataset Infos from /home/vashistt/.cache/huggingface/modules/datasets_modules/datasets/hellaswag/512a66dd8b1b1643ab4a48aa4f150d04c91680da6a4096498a5e5f799623d5ae
  Overwrite dataset info from restored data version if exists.
  03/29/2024 00:05:52 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/hellaswag/default/0.1.0/512a66dd8b1b1643ab4a48aa4f150d04c91680da6a4096498a5e5f799623d5ae
  03/29/2024 00:05:52 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/hellaswag/default/0.1.0/512a66dd8b1b1643ab4a48aa4f150d04c91680da6a4096498a5e5f799623d5ae
  Found cached dataset hellaswag (/home/vashistt/.cache/huggingface/datasets/hellaswag/default/0.1.0/512a66dd8b1b1643ab4a48aa4f150d04c91680da6a4096498a5e5f799623d5ae)
  03/29/2024 00:05:52 - INFO - datasets.builder - Found cached dataset hellaswag (/home/vashistt/.cache/huggingface/datasets/hellaswag/default/0.1.0/512a66dd8b1b1643ab4a48aa4f150d04c91680da6a4096498a5e5f799623d5ae)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/hellaswag/default/0.1.0/512a66dd8b1b1643ab4a48aa4f150d04c91680da6a4096498a5e5f799623d5ae
  03/29/2024 00:05:52 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/hellaswag/default/0.1.0/512a66dd8b1b1643ab4a48aa4f150d04c91680da6a4096498a5e5f799623d5ae
  Task: winogrande; number of docs: 1267
  Task: winogrande; document 0; context prompt (starting on next line):
  People think Rebecca
  (end of prompt on previous line)
  Requests: [Req_loglikelihood('People think Samantha', ' is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.')[0]
  , Req_loglikelihood('People think Rebecca', ' is embarassed, because Samantha made snide comments about the shirt Rebecca was wearing.')[0]
  ]
  Task: boolq; number of docs: 3270
  Task: boolq; document 0; context prompt (starting on next line):
  NCIS: New Orleans (season 4) -- The fourth season of NCIS: New Orleans premiered on September 26, 2017 on CBS. The series continues to air following Bull, Tuesday at 10:00 p.m. (ET) and contained 24 episodes. The season concluded on May 15, 2018.
  Question: is ncis new orleans over for the season?
  Answer:
  (end of prompt on previous line)
  Requests: (Req_loglikelihood('NCIS: New Orleans (season 4) -- The fourth season of NCIS: New Orleans premiered on September 26, 2017 on CBS. The series continues to air following Bull, Tuesday at 10:00 p.m. (ET) and contained 24 episodes. The season concluded on May 15, 2018.\nQuestion: is ncis new orleans over for the season?\nAnswer:', ' yes')[0]
  , Req_loglikelihood('NCIS: New Orleans (season 4) -- The fourth season of NCIS: New Orleans premiered on September 26, 2017 on CBS. The series continues to air following Bull, Tuesday at 10:00 p.m. (ET) and contained 24 episodes. The season concluded on May 15, 2018.\nQuestion: is ncis new orleans over for the season?\nAnswer:', ' no')[0]
  )
  Task: arc_challenge; number of docs: 1172
  Task: arc_challenge; document 0; context prompt (starting on next line):
  Question: Cities control the amount of pollution that is allowed to come from cars. How does this most likely help people?
  Answer:
  (end of prompt on previous line)
  Requests: [Req_loglikelihood('Question: Cities control the amount of pollution that is allowed to come from cars. How does this most likely help people?\nAnswer:', ' The air stays cleaner.')[0]
  , Req_loglikelihood('Question: Cities control the amount of pollution that is allowed to come from cars. How does this most likely help people?\nAnswer:', ' Cars can travel at faster speeds.')[0]
  , Req_loglikelihood('Question: Cities control the amount of pollution that is allowed to come from cars. How does this most likely help people?\nAnswer:', ' The skills of the drivers improve.')[0]
  , Req_loglikelihood('Question: Cities control the amount of pollution that is allowed to come from cars. How does this most likely help people?\nAnswer:', ' It becomes safer to drive on the roads.')[0]
  ]
  Task: arc_easy; number of docs: 2376
  Task: arc_easy; document 0; context prompt (starting on next line):
  Question: Which is the function of the gallbladder?
  Answer:
  (end of prompt on previous line)
  Requests: [Req_loglikelihood('Question: Which is the function of the gallbladder?\nAnswer:', ' store bile')[0]
  , Req_loglikelihood('Question: Which is the function of the gallbladder?\nAnswer:', ' produce bile')[0]
  , Req_loglikelihood('Question: Which is the function of the gallbladder?\nAnswer:', ' store digestive enzymes')[0]
  , Req_loglikelihood('Question: Which is the function of the gallbladder?\nAnswer:', ' produce digestive enzymes')[0]
  ]
  Task: hellaswag; number of docs: 10042
  Task: hellaswag; document 0; context prompt (starting on next line):
  Personal Care and Style: How to increase breast size with a bra. Check your bra size. Wearing a bra that is too big will not make your breasts look larger. That is why it is important to wear the right size bra for you.
  (end of prompt on previous line)
  Requests: [Req_loglikelihood('Personal Care and Style: How to increase breast size with a bra. Check your bra size. Wearing a bra that is too big will not make your breasts look larger. That is why it is important to wear the right size bra for you.', ' You can visit a lingerie shop and have them measure you to help you fit a bra to your size, or measure yourself before you shop for a new bra to ensure that you get a good fit. Use a flexible tape measure, like one found in a sewing kit.')[0]
  , Req_loglikelihood('Personal Care and Style: How to increase breast size with a bra. Check your bra size. Wearing a bra that is too big will not make your breasts look larger. That is why it is important to wear the right size bra for you.', ' This is why it is important to keep your breasts under protection when in the shower and only wear bras that are larger than your breast size. If you are not wearing a bra, try wearing something that is a little bigger.')[0]
  , Req_loglikelihood('Personal Care and Style: How to increase breast size with a bra. Check your bra size. Wearing a bra that is too big will not make your breasts look larger. That is why it is important to wear the right size bra for you.', ' For a girl, a bra with a support strap will be easier for her, because most women are unable to pull through bra straps and bras that are too small will not be able to support breasts from side-to-side. Many bras have even been created that cover the breast side, and can be sent to other women in the world to make them look bigger.')[0]
  , Req_loglikelihood('Personal Care and Style: How to increase breast size with a bra. Check your bra size. Wearing a bra that is too big will not make your breasts look larger. That is why it is important to wear the right size bra for you.', ' Choose a color that is flattering to your breast type and specific event, in addition to those that make you uncomfortable. Look for sports bras made from natural material, such as spandex or lycra, as this is a more breathable bra.')[0]
  ]
  Running loglikelihood requests
  100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 10127/10127 [06:13<00:00, 27.13it/s]
  {'results': {'winogrande': {'acc': 0.5213270142180095, 'acc_stderr': 0.019870831242787552}, 'boolq': {'acc': 0.5165876777251185, 'acc_stderr': 0.019877984170032503}, 'arc_challenge': {'acc': 0.19747235387045814, 'acc_stderr': 0.01583523867031247, 'acc_norm': 0.2559241706161137, 'acc_norm_stderr': 0.017358240916250527}, 'arc_easy': {'acc': 0.26540284360189575, 'acc_stderr': 0.01756381557124669, 'acc_norm': 0.2859399684044234, 'acc_norm_stderr': 0.017974062838604622}, 'hellaswag': {'acc': 0.29067930489731436, 'acc_stderr': 0.018062166146085863, 'acc_norm': 0.31121642969984203, 'acc_norm_stderr': 0.018416797316405352}}}
'''



# 2. Sheared llama


```
  (prune_llm_2) [vashistt@lovelace lora_ft]$ CUDA_VISIBLE_DEVICES=6 python3 Run_evals.py  --model_name_or_path "princeton-nlp/Sheared-LLaMA-2.7B"       
    --config_name "princeton-nlp/Sheared-LLaMA-2.7B"        --num_train_epochs 1         --block_size 512        --lora_r 128    --learning_rate 1e-4   
          --lora_alpha_ratio 4    --per_device_train_batch_size 1         --per_device_eval_batch_size 8       --do_train      --do_eval       --max_tr
  ain_samples 15000       --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"     --hidden_mse_w
  eight 0.0         --kl_weight 0.01        --dataset_name "wikitext"
  03/28/2024 22:39:16 - WARNING - __main__ - Process rank: -1, device: cuda:0, n_gpu: 1distributed training: False, 16-bits training: False
  03/28/2024 22:39:16 - INFO - __main__ - Training/evaluation parameters TrainingArguments(
  _n_gpu=1,
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
  dataloader_pin_memory=True,
  ddp_bucket_cap_mb=None,
  ddp_find_unused_parameters=None,
  ddp_timeout=1800,
  debug=[],
  deepspeed=None,
  disable_tqdm=False,
  do_eval=True,
  do_predict=False,
  do_train=True,
  eval_accumulation_steps=None,
  eval_delay=0,
  eval_steps=None,
  evaluation_strategy=no,
  fp16=False,
  fp16_backend=auto,
  fp16_full_eval=False,
  fp16_opt_level=O1,
  fsdp=[],
  fsdp_config={'fsdp_min_num_params': 0, 'xla': False, 'xla_fsdp_grad_ckpt': False},
  fsdp_min_num_params=0,
  fsdp_transformer_layer_cls_to_wrap=None,
  full_determinism=False,
  gradient_accumulation_steps=1,
  gradient_checkpointing=False,
  greater_is_better=None,
  group_by_length=False,
  half_precision_backend=auto,
  hub_model_id=None,
  hub_private_repo=False,
  hub_strategy=every_save,
  hub_token=<HUB_TOKEN>,
  ignore_data_skip=False,
  include_inputs_for_metrics=False,
  jit_mode_eval=False,
  label_names=None,
  label_smoothing_factor=0.0,
  learning_rate=0.0001,
  length_column_name=length,
  load_best_model_at_end=False,
  local_rank=-1,
  log_level=passive,
  log_level_replica=warning,
  log_on_each_node=True,
  logging_dir=/home/vashistt/anlp-project/finetuned_model/runs/Mar28_22-39-16_lovelace.ece.local.cmu.edu,
  logging_first_step=False,
  logging_nan_inf_filter=True,
  logging_steps=500,
  logging_strategy=steps,
  lr_scheduler_type=linear,
  max_grad_norm=1.0,
  max_steps=-1,
  metric_for_best_model=None,
  mp_parameters=,
  no_cuda=False,
  num_train_epochs=1.0,
  optim=adamw_hf,
  optim_args=None,
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
  save_safetensors=False,
  save_steps=500,
  save_strategy=steps,
  save_total_limit=None,
  seed=42,
  sharded_ddp=[],
  skip_memory_metrics=True,
  tf32=None,
  torch_compile=False,
  torch_compile_backend=None,
  torch_compile_mode=None,
  torchdynamo=None,
  tpu_metrics_debug=False,
  tpu_num_cores=None,
  use_ipex=False,
  use_legacy_prediction_loop=False,
  use_mps_device=False,
  warmup_ratio=0.0,
  warmup_steps=0,
  weight_decay=0.0,
  xpu_backend=None,
  )
  Using custom data configuration default-a3e66ef7800043cd
  03/28/2024 22:39:22 - INFO - datasets.builder - Using custom data configuration default-a3e66ef7800043cd
  Loading Dataset Infos from /home/vashistt/ENTER/envs/prune_llm_2/lib/python3.9/site-packages/datasets/packaged_modules/json
  03/28/2024 22:39:22 - INFO - datasets.info - Loading Dataset Infos from /home/vashistt/ENTER/envs/prune_llm_2/lib/python3.9/site-packages/datasets/packaged_modules/json
  Overwrite dataset info from restored data version if exists.
  03/28/2024 22:39:22 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
  03/28/2024 22:39:22 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
  Found cached dataset c4 (/home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2)
  03/28/2024 22:39:22 - INFO - datasets.builder - Found cached dataset c4 (/home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
  03/28/2024 22:39:22 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/allenai___c4/default-a3e66ef7800043cd/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2
  /home/vashistt/ENTER/envs/prune_llm_2/lib/python3.9/site-packages/datasets/load.py:2516: FutureWarning: 'use_auth_token' was deprecated in favor of 'token' in version 2.14.0 and will be removed in 3.0.0.
  You can remove this warning by passing 'token=<use_auth_token>' instead.
    warnings.warn(
  Overwrite dataset info from restored data version if exists.
  03/28/2024 22:39:24 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/28/2024 22:39:24 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  03/28/2024 22:39:24 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/28/2024 22:39:24 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  Overwrite dataset info from restored data version if exists.
  03/28/2024 22:39:27 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/28/2024 22:39:27 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  03/28/2024 22:39:27 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/28/2024 22:39:27 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  [INFO|configuration_utils.py:668] 2024-03-28 22:39:27,092 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--princeton-nlp--Sheared-LLaMA-2.7B/snapshots/2f157a0306b75d37694ae05f6a4067220254d540/config.json
  [INFO|configuration_utils.py:720] 2024-03-28 22:39:27,093 >> Model config LlamaConfig {
    "_name_or_path": "princeton-nlp/Sheared-LLaMA-2.7B",
    "architectures": [
      "LlamaForCausalLM"
    ],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 6912,
    "max_position_embeddings": 4096,
    "model_type": "llama",
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
    "num_key_value_heads": 20,
    "pad_token_id": 0,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "tie_word_embeddings": false,
    "torch_dtype": "float32",
    "transformers_version": "4.28.0",
    "use_cache": true,
    "vocab_size": 32000
  }

  [INFO|tokenization_utils_base.py:1809] 2024-03-28 22:39:27,127 >> loading file tokenizer.model from cache at /home/vashistt/.cache/huggingface/hub/models--princeton-nlp--Sheared-LLaMA-2.7B/snapshots/2f157a0306b75d37694ae05f6a4067220254d540/tokenizer.model
  [INFO|tokenization_utils_base.py:1809] 2024-03-28 22:39:27,128 >> loading file added_tokens.json from cache at None
  [INFO|tokenization_utils_base.py:1809] 2024-03-28 22:39:27,128 >> loading file special_tokens_map.json from cache at /home/vashistt/.cache/huggingface/hub/models--princeton-nlp--Sheared-LLaMA-2.7B/snapshots/2f157a0306b75d37694ae05f6a4067220254d540/special_tokens_map.json
  [INFO|tokenization_utils_base.py:1809] 2024-03-28 22:39:27,128 >> loading file tokenizer_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--princeton-nlp--Sheared-LLaMA-2.7B/snapshots/2f157a0306b75d37694ae05f6a4067220254d540/tokenizer_config.json
  [INFO|configuration_utils.py:668] 2024-03-28 22:39:27,196 >> loading configuration file config.json from cache at /home/vashistt/.cache/huggingface/hub/models--princeton-nlp--Sheared-LLaMA-2.7B/snapshots/2f157a0306b75d37694ae05f6a4067220254d540/config.json
  [INFO|configuration_utils.py:720] 2024-03-28 22:39:27,197 >> Model config LlamaConfig {
    "_name_or_path": "princeton-nlp/Sheared-LLaMA-2.7B",
    "architectures": [
      "LlamaForCausalLM"
    ],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 6912,
    "max_position_embeddings": 4096,
    "model_type": "llama",
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
    "num_key_value_heads": 20,
    "pad_token_id": 0,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "tie_word_embeddings": false,
    "torch_dtype": "float16",
    "transformers_version": "4.28.0",
    "use_cache": true,
    "vocab_size": 32000
  }

  [INFO|modeling_utils.py:2534] 2024-03-28 22:39:27,200 >> loading weights file pytorch_model.bin from cache at /home/vashistt/.cache/huggingface/hub/models--princeton-nlp--Sheared-LLaMA-2.7B/snapshots/2f157a0306b75d37694ae05f6a4067220254d540/pytorch_model.bin.index.json
  [INFO|modeling_utils.py:1176] 2024-03-28 22:39:27,200 >> Instantiating LlamaForCausalLM model under default dtype torch.float16.
  [INFO|configuration_utils.py:575] 2024-03-28 22:39:27,201 >> Generate config GenerationConfig {
    "_from_model_config": true,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "transformers_version": "4.28.0"
  }

  Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:05<00:00,  2.67s/it]
  [INFO|modeling_utils.py:3190] 2024-03-28 22:39:33,371 >> All model checkpoint weights were used when initializing LlamaForCausalLM.

  [INFO|modeling_utils.py:3198] 2024-03-28 22:39:33,372 >> All the weights of LlamaForCausalLM were initialized from the model checkpoint at princeton-nlp/Sheared-LLaMA-2.7B.
  If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
  [INFO|configuration_utils.py:537] 2024-03-28 22:39:33,415 >> loading configuration file generation_config.json from cache at /home/vashistt/.cache/huggingface/hub/models--princeton-nlp--Sheared-LLaMA-2.7B/snapshots/2f157a0306b75d37694ae05f6a4067220254d540/generation_config.json
  [INFO|configuration_utils.py:575] 2024-03-28 22:39:33,416 >> Generate config GenerationConfig {
    "_from_model_config": true,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
    "transformers_version": "4.28.0"
  }

  Num params = :  2537720320
  03/28/2024 22:39:33 - INFO - __main__ - *** Evaluate ***
  Overwrite dataset info from restored data version if exists.
  03/28/2024 22:39:36 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/28/2024 22:39:36 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  03/28/2024 22:39:36 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/28/2024 22:39:36 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▋ | 83/84 [01:53<00:01,  1.37s/it]
  Original perplexity on wikitext = 6.414
  Num params = :  2537720320
  eleuther eval for original model
  original model param count : 2537720320
  epoch 1, param count is 2030476800
  epoch 2, param count is 1522936320
  epoch 3, param count is 1268866560
  Final model sparsity is : 0.500 
  Final model param count : 1268866560
  Num params = :  1268866560
  03/28/2024 22:42:18 - INFO - __main__ - *** Evaluate ***
  Overwrite dataset info from restored data version if exists.
  03/28/2024 22:42:21 - INFO - datasets.builder - Overwrite dataset info from restored data version if exists.
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/28/2024 22:42:21 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  03/28/2024 22:42:21 - INFO - datasets.builder - Found cached dataset wikitext (/home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3)
  Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  03/28/2024 22:42:21 - INFO - datasets.info - Loading Dataset info from /home/vashistt/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3
  99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▋ | 83/84 [00:32<00:00,  2.55it/s]
  [SpeedUp=3.668] Original perplexity on wikitext = 6.414 | Before Training perplexity on wikitext = 94.688
'''


