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

    '''Traceback (most recent call last):
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

'''You are using a model of type phi to instantiate a model of type llama. This is not supported for all configurations of models and can yield errors.'''
