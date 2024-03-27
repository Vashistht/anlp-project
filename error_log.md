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
