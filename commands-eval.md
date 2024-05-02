- eval on ppl (`--do_eval` remove it otherwise)
- `--do_eleuther_eval_og_model True` (if u want stats on the og model)
- `--add_finetuned_adapter True`: if you want to calculate stats for different sparsity than the final one + whether to use finetuned or not 
- `--prune_target_epoch 3` (3 means .5 sparsity, 2 for .4, and 1 for .2) ideally want to run for each of them all else same

```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval True  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter True --do_eleuther_eval_og_model True --do_eval > prune_ds_MetricWeights_FT_epoch_{targetEpoch}.txt
```

<!-- # 1. wikitext pruned masks from sheared 
location="/home/vashistt/Desktop/anlp-project/outdir_sheared_llama-a3/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=200_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-sheared-llama2-wikitext_Adaptive=Yes"


```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals_math-qa.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval True  --dataset_name 'gsm8k' --prune_target_epoch 2 --add_finetuned_adapter False --do_eleuther_eval_og_model False --save_info 'pruned_(sheared_2)_wikitext_metrics_ppl' > pruned_gsm8k_metrics_sheared_math.txt 2>&1
``` -->

# 2. wikitext on llama masks from a3

location="/home/vashistt/Desktop/anlp-project/outdir_llama_2_7b-a3/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2-wikitext_Adaptive=Yes"

```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals_math-qa.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval True  --dataset_name 'wiki' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --save_info 'pruned_wikitext_metrics_ppl' > pruned_wiki_math.txt 2>&1
```

# 3. c4
prune_info='c4_masks-100'
outdir="/home/vashistt/Desktop/anlp-project/finetuned_model/${prune_info}"

location="/home/vashistt/Desktop/anlp-project/outdir_c4/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2_Adaptive=Yes"

```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals_math-qa.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval True  --dataset_name 'c4' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --save_info 'pruned_c4_metrics_ppl' > pruned_c4_math.txt 2>&1
```


# 4. acc (20)
prune_info='acc_masks-100'
outdir="/home/vashistt/Desktop/anlp-project/finetuned_model/${prune_info}"

location="/home/vashistt/Desktop/anlp-project/outdir_gsm8k_a4_acc/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2-acc-only_Adaptive=Yes"

```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals_math-qa.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval True  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --save_info 'pruned_gsm8k_metrics_acc' > pruned_gsm8k_acc_math.txt 2>&1
```

--- 
Now adding gsm8k as well 
# 5. (33.333x3) 
prune_info='0_ppl_3each_masks-100'
outdir="/home/vashistt/Desktop/anlp-project/finetuned_model/${prune_info}"
location="/home/vashistt/Desktop/anlp-project/outdir_gsm8k_a4_100toks/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2_Adaptive=Yes"


```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals_math-qa.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval True  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --save_info 'pruned_gsm8k_metrics_3each' > pruned_gsm8k_3each_math.txt 2>&1
```

```
CUDA_VISIBLE_DEVICES=6 python3 Run_evals_non_math.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval True  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --save_info 'pruned_gsm8k_metrics_3each' > pruned_gsm8k_3each_non-math.txt 2>&1
```
# 6. (0-50-50-0) 
prune_info='0_ppl_0-50-50-0_masks-100'
outdir="/home/vashistt/Desktop/anlp-project/finetuned_model/${prune_info}"
location="/home/vashistt/Desktop/anlp-project/outdir_gsm8k_a4_100toks_0-5050-0/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2_Adaptive=Yes"


```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals_math-qa.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval True  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --save_info 'pruned_gsm8k_metrics_0-50-50-0' > pruned_gsm8k_0-50-50-0_math.txt 2>&1
```

```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals_non_math.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval True  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --save_info 'pruned_gsm8k_metrics_0-50-50-0' > pruned_gsm8k_0-50-50-0_non-math_correct.txt 2>&1
```

# 7 (20) 38/24/38

prune_info='0_ppl_0-38-24-38-0_masks-100'
location='/home/vashistt/Desktop/anlp-project/outdir_gsm8k_a4_combined_382438/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2_Adaptive=Yes'
outdir="/home/vashistt/Desktop/anlp-project/finetuned_model/${prune_info}"

```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals_math-qa.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval True  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --save_info 'pruned_gsm8k_metrics_0-38-24-38' > pruned_gsm8k_0-38-24-38_math.txt 2>&1
```


# 8 gsm8k (ppl)--run1-with2hints
prune_info='ppl_gsm8k' 
location='/home/vashistt/Desktop/anlp-project/outdir_gsm8k/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2_Adaptive=Yes'
outdir="/home/vashistt/Desktop/anlp-project/finetuned_model/${prune_info}"

```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals_math-qa.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval True  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --save_info 'pruned_gsm8k_metrics_ppl' > pruned_gsm8k_pplmath.txt 2>&1
```

```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals_non_math.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval True  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --save_info 'pruned_gsm8k_metrics_ppl' > pruned_gsm8k_ppl_no_math.txt 2>&1
```


# 9 gsm8k (hybrid)
```
    ppl_term_train = ppl_weight * ppl_train
	other_terms = lexsim_weight * lexsim_train + cossim_weight * cossim_train + acc_weight * acc_train
	if (ppl_term_train)> (other_terms):
		combined_train =  -1*ppl_term_train
	else:
		combined_train = other_terms
```

prune_info='hybrdid_gsm8k' 
location='/home/vashistt/Desktop/anlp-project/outdir_gsm8k_a4_hybrid/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2_Adaptive=Yes'
outdir="/home/vashistt/Desktop/anlp-project/finetuned_model/${prune_info}"

```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals_math-qa.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval True  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --save_info 'pruned_gsm8k_metrics_hybrid' > pruned_gsm8k_hybridmath.txt 2>&1
```

```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals_non_math.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval True  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --save_info 'pruned_gsm8k_metrics_hybrid' > pruned_gsm8k_hybrid_no_math.txt 2>&1
```



# 8 gsm8k (ppl) -- no hints (200 masks)
prune_info='ppl_gsm8k-200-mask' 
location='/home/vashistt/Desktop/anlp-project/outdir_gsm8k_a4_ppl/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=200_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2_Adaptive=Yes'
outdir="/home/vashistt/Desktop/anlp-project/finetuned_model/${prune_info}"

```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals_math-qa.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval True  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --save_info 'pruned_gsm8k_metrics_ppl_200masks' > pruned_gsm8k_pplmath_200masks.txt 2>&1
```

```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals_non_math.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval True  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --save_info 'pruned_gsm8k_metrics_ppl_200masks' > pruned_gsm8k_ppl_no_math_200masks.txt 2>&1
```





---
# ppl on gsm 

1. wiki 
prune_info='wiki_masks-100'
outdir="/home/vashistt/Desktop/anlp-project/finetuned_model/${prune_info}"
location="/home/vashistt/Desktop/anlp-project/outdir_llama_2_7b-a3/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2-wikitext_Adaptive=Yes"

```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals_math-qa.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval False  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --do_eval --save_info 'pruned_wikitext_metrics_ppl_gsmppl' > pruned_wiki_gsmppl.txt 2>&1
```

2. emily split
prune_info='gsm_masks-38-24-28'
outdir="/home/vashistt/Desktop/anlp-project/finetuned_model/${prune_info}"
location="/home/vashistt/Desktop/anlp-project/outdir_gsm8k_a4_combined_382438/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2_Adaptive=Yes"

```
CUDA_VISIBLE_DEVICES=6 python3 Run_evals_math-qa.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval False  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --do_eval --save_info 'pruned_gsm-38-24-28_metrics_ppl_gsmppl' > pruned_gsm-38-24-28_gsmppl.txt 2>&1
```

3. gsm (acc)
prune_info='gsm_masks-100acc'
outdir="/home/vashistt/Desktop/anlp-project/outdir_gsm8k_a4_acc/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2-acc-only_Adaptive=Yes"

```
CUDA_VISIBLE_DEVICES=7 python3 Run_evals_math-qa.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval False  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --do_eval --save_info 'pruned_gsm-100acc_gsmppl' > pruned_100acc_gsmppl.txt 2>&1
```

4. 33-33-33 split
prune_info='gsm_masks-33-33-33'
outdir="/home/vashistt/Desktop/anlp-project/outdir_gsm8k_a4_100toks/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2_Adaptive=Yes"

```
CUDA_VISIBLE_DEVICES=9 python3 Run_evals_math-qa.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval False  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --do_eval --save_info 'pruned_gsm-33-33-33_metrics_ppl_gsmppl' > pruned_gsm-33-33-33_gsmppl.txt 2>&1
```

4. 0-50-50 split
prune_info='gsm_masks-0-50-50'
outdir="/home/vashistt/Desktop/anlp-project/outdir_gsm8k_a4_100toks_0-5050-0/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2_Adaptive=Yes"

```
CUDA_VISIBLE_DEVICES=9 python3 Run_evals_math-qa.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512    --learning_rate 1e-4               --per_device_train_batch_size 1         --per_device_eval_batch_size 8  --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"   --do_eleuther_eval False  --dataset_name 'gsm8k' --prune_target_epoch 3 --add_finetuned_adapter False --do_eleuther_eval_og_model False --do_eval --save_info 'pruned_gsm-0-50-50_metrics_ppl_gsmppl' > pruned_gsm-0-50-50_gsmppl.txt 2>&1
```
