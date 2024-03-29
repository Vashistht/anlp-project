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
```

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
```

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
```


## GSM-8k (5 shot)
```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512        --lora_r 128    --learning_rate 1e-4            --lora_alpha_ratio 4    --per_device_train_batch_size 1         --per_device_eval_batch_size 8       --do_train      --do_eval       --max_train_samples 15000       --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"     --hidden_mse_weight 0.0         --kl_weight 0.01        --dataset_name "wikitext"
```

(forgot to copy the one after the command, but this runs for a long long time :c )

```
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 70%|████████████████████████████████████████████████████████████████████████████████████████████████████▉                                            | 459/659 [1:05:15<28:09,  8.45s/it][INFO|configuration_utils.py:575] 2024-03-29 01:44:52,182 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 70%|█████████████████████████████████████████████████████████████████████████████████████████████████████▏                                           | 460/659 [1:05:23<28:02,  8.46s/it][INFO|configuration_utils.py:575] 2024-03-29 01:45:00,664 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 70%|█████████████████████████████████████████████████████████████████████████████████████████████████████▍                                           | 461/659 [1:05:32<27:55,  8.46s/it][INFO|configuration_utils.py:575] 2024-03-29 01:45:09,133 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 70%|█████████████████████████████████████████████████████████████████████████████████████████████████████▋                                           | 462/659 [1:05:40<27:45,  8.45s/it][INFO|configuration_utils.py:575] 2024-03-29 01:45:17,567 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 70%|█████████████████████████████████████████████████████████████████████████████████████████████████████▊                                           | 463/659 [1:05:49<27:35,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 01:45:25,990 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 70%|██████████████████████████████████████████████████████████████████████████████████████████████████████                                           | 464/659 [1:05:57<27:25,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 01:45:34,411 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 71%|██████████████████████████████████████████████████████████████████████████████████████████████████████▎                                          | 465/659 [1:06:05<27:17,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 01:45:42,855 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 71%|██████████████████████████████████████████████████████████████████████████████████████████████████████▌                                          | 466/659 [1:06:14<27:10,  8.45s/it][INFO|configuration_utils.py:575] 2024-03-29 01:45:51,317 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 71%|██████████████████████████████████████████████████████████████████████████████████████████████████████▊                                          | 467/659 [1:06:22<27:01,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 01:45:59,756 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 71%|██████████████████████████████████████████████████████████████████████████████████████████████████████▉                                          | 468/659 [1:06:31<26:51,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 01:46:08,186 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 71%|███████████████████████████████████████████████████████████████████████████████████████████████████████▏                                         | 469/659 [1:06:39<26:41,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:46:16,589 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 71%|███████████████████████████████████████████████████████████████████████████████████████████████████████▍                                         | 470/659 [1:06:48<26:31,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 01:46:24,996 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 71%|███████████████████████████████████████████████████████████████████████████████████████████████████████▋                                         | 471/659 [1:06:56<26:24,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:46:33,444 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 72%|███████████████████████████████████████████████████████████████████████████████████████████████████████▊                                         | 472/659 [1:07:04<26:19,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 01:46:41,922 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 72%|████████████████████████████████████████████████████████████████████████████████████████████████████████                                         | 473/659 [1:07:13<26:11,  8.45s/it][INFO|configuration_utils.py:575] 2024-03-29 01:46:50,373 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 72%|████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                        | 474/659 [1:07:21<26:01,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 01:46:58,802 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 72%|████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                        | 475/659 [1:07:30<25:52,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 01:47:07,225 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 72%|████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                        | 476/659 [1:07:38<25:43,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:47:15,651 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 72%|████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                        | 477/659 [1:07:47<25:34,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:47:24,085 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 73%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                       | 478/659 [1:07:55<25:26,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:47:32,521 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 73%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                       | 479/659 [1:08:04<25:18,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 01:47:40,959 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 73%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                       | 480/659 [1:08:12<25:09,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:47:49,378 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 73%|█████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                       | 481/659 [1:08:20<24:59,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:47:57,792 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 73%|██████████████████████████████████████████████████████████████████████████████████████████████████████████                                       | 482/659 [1:08:29<24:51,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 01:48:06,213 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 73%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                      | 483/659 [1:08:37<24:43,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:48:14,659 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 73%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                      | 484/659 [1:08:46<24:35,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:48:23,086 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 74%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                      | 485/659 [1:08:54<24:27,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:48:31,521 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 74%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                      | 486/659 [1:09:03<24:18,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:48:39,951 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 74%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                     | 487/659 [1:09:11<24:10,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:48:48,382 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 74%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                     | 488/659 [1:09:19<24:01,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:48:56,809 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 74%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                     | 489/659 [1:09:28<23:53,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:49:05,251 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 74%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                     | 490/659 [1:09:36<23:46,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 01:49:13,702 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 75%|████████████████████████████████████████████████████████████████████████████████████████████████████████████                                     | 491/659 [1:09:45<23:37,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:49:22,128 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 75%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                    | 492/659 [1:09:53<23:28,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:49:30,561 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 75%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                    | 493/659 [1:10:02<23:21,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 01:49:39,019 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 75%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                    | 494/659 [1:10:10<23:11,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 01:49:47,439 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 75%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                    | 495/659 [1:10:18<23:03,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:49:55,871 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 75%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                   | 496/659 [1:10:27<22:54,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:50:04,300 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 75%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                   | 497/659 [1:10:35<22:45,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:50:12,730 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 76%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                   | 498/659 [1:10:44<22:37,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:50:21,161 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 76%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                   | 499/659 [1:10:52<22:29,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:50:29,593 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 76%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████                                   | 500/659 [1:11:01<22:19,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:50:38,010 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 76%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                  | 501/659 [1:11:09<22:12,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:50:46,458 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 76%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                  | 502/659 [1:11:17<22:03,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:50:54,885 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 76%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                  | 503/659 [1:11:26<21:54,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 01:51:03,298 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 76%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                  | 504/659 [1:11:34<21:44,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 01:51:11,700 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 77%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████                                  | 505/659 [1:11:43<21:35,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 01:51:20,087 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 77%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                 | 506/659 [1:11:51<21:25,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 01:51:28,464 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 77%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                 | 507/659 [1:11:59<21:17,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 01:51:36,880 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 77%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                 | 508/659 [1:12:08<21:09,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 01:51:45,299 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 77%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                 | 509/659 [1:12:16<21:00,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 01:51:53,696 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 77%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                | 510/659 [1:12:25<20:50,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:52:02,062 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 78%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                | 511/659 [1:12:33<20:41,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:52:10,446 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 78%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                | 512/659 [1:12:41<20:32,  8.38s/it][INFO|configuration_utils.py:575] 2024-03-29 01:52:18,813 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 78%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                | 513/659 [1:12:50<20:25,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:52:27,228 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 78%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                | 514/659 [1:12:58<20:17,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:52:35,621 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 78%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                               | 515/659 [1:13:07<20:08,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:52:44,001 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 78%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                               | 516/659 [1:13:15<19:59,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:52:52,381 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 78%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                               | 517/659 [1:13:23<19:50,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:53:00,766 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 79%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                               | 518/659 [1:13:32<19:42,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:53:09,150 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 79%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                              | 519/659 [1:13:40<19:35,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 01:53:17,569 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 79%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                              | 520/659 [1:13:49<19:26,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 01:53:25,963 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 79%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                              | 521/659 [1:13:57<19:17,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:53:34,334 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 79%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                              | 522/659 [1:14:05<19:07,  8.38s/it][INFO|configuration_utils.py:575] 2024-03-29 01:53:42,686 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 79%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████                              | 523/659 [1:14:14<18:59,  8.38s/it][INFO|configuration_utils.py:575] 2024-03-29 01:53:51,061 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 80%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                             | 524/659 [1:14:22<18:51,  8.38s/it][INFO|configuration_utils.py:575] 2024-03-29 01:53:59,457 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 80%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                             | 525/659 [1:14:30<18:43,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:54:07,854 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 80%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                             | 526/659 [1:14:39<18:36,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:54:16,261 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 80%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                             | 527/659 [1:14:47<18:28,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 01:54:24,682 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 80%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                            | 528/659 [1:14:56<18:19,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 01:54:33,067 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 80%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                            | 529/659 [1:15:04<18:11,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:54:41,457 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 80%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                            | 530/659 [1:15:12<18:02,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:54:49,829 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 81%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                            | 531/659 [1:15:21<17:54,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:54:58,230 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 81%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                            | 532/659 [1:15:29<17:45,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:55:06,621 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 81%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                           | 533/659 [1:15:38<17:37,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:55:15,005 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 81%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                           | 534/659 [1:15:46<17:28,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:55:23,400 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 81%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                           | 535/659 [1:15:54<17:19,  8.38s/it][INFO|configuration_utils.py:575] 2024-03-29 01:55:31,770 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 81%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                           | 536/659 [1:16:03<17:10,  8.38s/it][INFO|configuration_utils.py:575] 2024-03-29 01:55:40,142 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 81%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                          | 537/659 [1:16:11<17:03,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:55:48,564 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 82%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                          | 538/659 [1:16:20<16:56,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 01:55:56,983 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 82%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                          | 539/659 [1:16:28<16:48,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 01:56:05,393 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 82%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                          | 540/659 [1:16:36<16:39,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 01:56:13,788 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 82%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                          | 541/659 [1:16:45<16:30,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:56:22,160 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 82%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                         | 542/659 [1:16:53<16:21,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:56:30,528 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 82%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                         | 543/659 [1:17:01<16:12,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:56:38,921 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 83%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                         | 544/659 [1:17:10<16:04,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:56:47,310 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 83%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                         | 545/659 [1:17:18<15:56,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:56:55,708 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 83%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                        | 546/659 [1:17:27<15:48,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:57:04,099 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 83%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                        | 547/659 [1:17:35<15:39,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:57:12,482 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 83%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                        | 548/659 [1:17:43<15:30,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:57:20,862 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 83%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                        | 549/659 [1:17:52<15:23,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 01:57:29,289 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 83%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                        | 550/659 [1:18:00<15:15,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 01:57:37,688 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                       | 551/659 [1:18:09<15:06,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:57:46,067 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                       | 552/659 [1:18:17<14:57,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:57:54,457 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                       | 553/659 [1:18:25<14:49,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:58:02,844 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 84%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                       | 554/659 [1:18:34<14:41,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:58:11,237 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 84%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                       | 555/659 [1:18:42<14:32,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:58:19,633 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 84%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                      | 556/659 [1:18:51<14:25,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 01:58:28,044 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                      | 557/659 [1:18:59<14:17,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 01:58:36,458 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                      | 558/659 [1:19:07<14:08,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 01:58:44,852 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 85%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                      | 559/659 [1:19:16<13:59,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:58:53,233 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 85%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                     | 560/659 [1:19:24<13:50,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:59:01,618 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 85%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                     | 561/659 [1:19:33<13:42,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:59:10,019 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 85%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                     | 562/659 [1:19:41<13:33,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:59:18,399 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 85%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                     | 563/659 [1:19:49<13:24,  8.38s/it][INFO|configuration_utils.py:575] 2024-03-29 01:59:26,755 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 86%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                     | 564/659 [1:19:58<13:15,  8.38s/it][INFO|configuration_utils.py:575] 2024-03-29 01:59:35,124 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 86%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                    | 565/659 [1:20:06<13:07,  8.38s/it][INFO|configuration_utils.py:575] 2024-03-29 01:59:43,509 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 86%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                    | 566/659 [1:20:14<12:59,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 01:59:51,910 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 86%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                    | 567/659 [1:20:23<12:52,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:00:00,332 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 86%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                    | 568/659 [1:20:31<12:45,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:00:08,765 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 86%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                   | 569/659 [1:20:40<12:36,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:00:17,164 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 86%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                   | 570/659 [1:20:48<12:27,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:00:25,561 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 87%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                   | 571/659 [1:20:57<12:19,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:00:33,974 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 87%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                   | 572/659 [1:21:05<12:10,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:00:42,368 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 87%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                   | 573/659 [1:21:13<12:02,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:00:50,772 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 88%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                | 583/659 [1:22:37<10:38,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:02:14,776 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 88%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                 | 579/659 [1:22:04<11:12,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:01:41,140 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 88%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                 | 580/659 [1:22:12<11:04,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:01:49,567 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 88%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                 | 581/659 [1:22:21<10:56,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:01:57,976 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 88%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                 | 582/659 [1:22:29<10:47,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:02:06,379 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 88%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                | 583/659 [1:22:37<10:38,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:02:14,776 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 89%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                | 584/659 [1:22:46<10:29,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:02:23,158 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 89%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                | 585/659 [1:22:54<10:22,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:02:31,595 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 89%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                | 586/659 [1:23:03<10:14,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 02:02:40,026 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 89%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏               | 587/659 [1:23:11<10:05,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:02:48,433 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 89%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍               | 588/659 [1:23:19<09:57,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:02:56,839 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 89%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌               | 589/659 [1:23:28<09:48,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:03:05,222 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 90%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊               | 590/659 [1:23:36<09:39,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:03:13,604 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████               | 591/659 [1:23:45<09:31,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:03:22,016 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎              | 592/659 [1:23:53<09:23,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:03:30,428 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍              | 593/659 [1:24:01<09:14,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:03:38,819 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋              | 594/659 [1:24:10<09:06,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:03:47,225 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 90%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉              | 595/659 [1:24:18<08:57,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:03:55,620 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 90%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏             | 596/659 [1:24:27<08:49,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:04:04,032 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 91%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎             | 597/659 [1:24:35<08:41,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 02:04:12,482 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 91%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌             | 598/659 [1:24:43<08:33,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 02:04:20,913 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 91%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊             | 599/659 [1:24:52<08:24,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 02:04:29,315 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 91%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████             | 600/659 [1:25:00<08:16,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:04:37,711 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 91%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏            | 601/659 [1:25:09<08:07,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:04:46,114 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 91%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍            | 602/659 [1:25:17<07:58,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:04:54,497 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 92%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋            | 603/659 [1:25:25<07:49,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 02:05:02,866 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 92%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉            | 604/659 [1:25:34<07:41,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:05:11,274 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████            | 605/659 [1:25:42<07:33,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:05:19,670 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎           | 606/659 [1:25:51<07:24,  8.39s/it][INFO|configuration_utils.py:575] 2024-03-29 02:05:28,061 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌           | 607/659 [1:25:59<07:16,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:05:36,461 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊           | 608/659 [1:26:07<07:08,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:05:44,862 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 92%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉           | 609/659 [1:26:16<07:00,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:05:53,291 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 93%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏          | 610/659 [1:26:24<06:52,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:06:01,710 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 93%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍          | 611/659 [1:26:33<06:43,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:06:10,129 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 93%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋          | 612/659 [1:26:41<06:35,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:06:18,530 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 93%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉          | 613/659 [1:26:49<06:26,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:06:26,919 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 93%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████          | 614/659 [1:26:58<06:17,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:06:35,300 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 93%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎         | 615/659 [1:27:06<06:09,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:06:43,715 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 93%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌         | 616/659 [1:27:15<06:01,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:06:52,140 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 94%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊         | 617/659 [1:27:23<05:53,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:07:00,548 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 94%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉         | 618/659 [1:27:32<05:44,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:07:08,946 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 94%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏        | 619/659 [1:27:40<05:36,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:07:17,360 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 94%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍        | 620/659 [1:27:48<05:27,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:07:25,768 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 94%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋        | 621/659 [1:27:57<05:19,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:07:34,183 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 94%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊        | 622/659 [1:28:05<05:11,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:07:42,606 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 95%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████        | 623/659 [1:28:14<05:02,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:07:51,016 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 95%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎       | 624/659 [1:28:22<04:54,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:07:59,404 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 95%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌       | 625/659 [1:28:30<04:45,  8.40s/it][INFO|configuration_utils.py:575] 2024-03-29 02:08:07,801 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 95%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋       | 626/659 [1:28:39<04:37,  8.41s/it][INFO|configuration_utils.py:575] 2024-03-29 02:08:16,215 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 95%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉       | 627/659 [1:28:47<04:29,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 02:08:24,657 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 95%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏      | 628/659 [1:28:56<04:21,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 02:08:33,098 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 95%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍      | 629/659 [1:29:04<04:12,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 02:08:41,497 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 96%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌      | 630/659 [1:29:12<04:04,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 02:08:49,915 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 96%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊      | 631/659 [1:29:21<03:55,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 02:08:58,338 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 96%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████      | 632/659 [1:29:29<03:47,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 02:09:06,751 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 96%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎     | 633/659 [1:29:38<03:39,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 02:09:15,217 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 96%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍     | 634/659 [1:29:46<03:30,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 02:09:23,648 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 96%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋     | 635/659 [1:29:55<03:22,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 02:09:32,064 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 97%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉     | 636/659 [1:30:03<03:13,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 02:09:40,486 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 97%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏    | 637/659 [1:30:11<03:05,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 02:09:48,903 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 97%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍    | 638/659 [1:30:20<02:56,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 02:09:57,323 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 97%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌    | 639/659 [1:30:28<02:48,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 02:10:05,780 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 97%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊    | 640/659 [1:30:37<02:40,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 02:10:14,220 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 97%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████    | 641/659 [1:30:45<02:31,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 02:10:22,630 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 97%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎   | 642/659 [1:30:54<02:23,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 02:10:31,034 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 98%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍   | 643/659 [1:31:02<02:14,  8.42s/it][INFO|configuration_utils.py:575] 2024-03-29 02:10:39,465 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 98%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋   | 644/659 [1:31:10<02:06,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 02:10:47,907 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 98%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉   | 645/659 [1:31:19<01:58,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 02:10:56,347 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏  | 646/659 [1:31:27<01:49,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 02:11:04,786 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎  | 647/659 [1:31:36<01:41,  8.43s/it][INFO|configuration_utils.py:575] 2024-03-29 02:11:13,211 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌  | 648/659 [1:31:44<01:32,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 02:11:21,678 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊  | 649/659 [1:31:53<01:24,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 02:11:30,110 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████  | 650/659 [1:32:01<01:16,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 02:11:38,568 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏ | 651/659 [1:32:10<01:07,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 02:11:47,007 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍ | 652/659 [1:32:18<00:59,  8.45s/it][INFO|configuration_utils.py:575] 2024-03-29 02:11:55,485 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋ | 653/659 [1:32:27<00:50,  8.46s/it][INFO|configuration_utils.py:575] 2024-03-29 02:12:03,961 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉ | 654/659 [1:32:35<00:42,  8.45s/it][INFO|configuration_utils.py:575] 2024-03-29 02:12:12,391 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

 99%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ | 655/659 [1:32:43<00:33,  8.44s/it][INFO|configuration_utils.py:575] 2024-03-29 02:12:20,791 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎| 656/659 [1:32:52<00:25,  8.48s/it][INFO|configuration_utils.py:575] 2024-03-29 02:12:29,360 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 657/659 [1:33:00<00:16,  8.47s/it][INFO|configuration_utils.py:575] 2024-03-29 02:12:37,821 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}


100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎| 656/659 [1:32:52<00:25,  8.48s/it][INFO|configuration_utils.py:575] 2024-03-29 02:12:29,360 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌| 657/659 [1:33:00<00:16,  8.47s/it][INFO|configuration_utils.py:575] 2024-03-29 02:12:37,821 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊| 658/659 [1:33:09<00:08,  8.53s/it][INFO|configuration_utils.py:575] 2024-03-29 02:12:46,488 >> Generate config GenerationConfig {
  "_from_model_config": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "transformers_version": "4.28.0"
}

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 659/659 [1:33:18<00:00,  8.49s/it]
{'results': {'gsm8k': {'acc': 0.0, 'acc_stderr': 0.0}}}
```


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
```

