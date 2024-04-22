# Setting up 

### Create and activate new conda environment

```
conda create -n prune_llm mamba python=3.9 -c conda-forge
source activate prune_llm

conda install -c pytorch -c conda-forge -c defaults wheel=0.41.2 # from the env yaml

conda install -c pytorch -c conda-forge -c defaults transformers==4.28.0 tokenizers==0.13.3 

conda install -c pytorch -c conda-forge -c defaults lxml==5.1.0

conda install -c pytorch -c conda-forge -c defaults peft==0.6.2

conda install -c pytorch -c conda-forge -c defaults  sentencepiece==0.1.99

pip3 install protobuf google

<!-- conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -->

pip3 install torch==2.1.1 

```
from yaml file
--- torchvision=0.11.2, torchaudio=0.10.1

### Install PyTorch, torchvision, torchaudio, and other libraries
```
pip3 install ipykernel scipy matplotlib
```

### Install additional Python packages
```
pip3 install torch numpy wandb accelerate==0.24.1 chardet==5.2.0 datasets==2.16.1 huggingface-hub pandas plotly tqdm urllib3 
```

### Upgrade huggingface_hub for CLI support
```
pip3 install -U "huggingface_hub[cli]"
```

# For finetuning
```
pip3 install evaluate==0.4.1 peft==0.6.2
```

# for lm-eval

```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

### Register environment with Jupyter
```
python -m ipykernel install --user --name=prune_llm
```
### Login to Hugging Face CLI
```
huggingface-cli login --token YOUR_HF_API_TOKEN
```

# Running the commands


## Prune LLaMA 2 7B
```
CUDA_VISIBLE_DEVICES=0 python3 main.py --model meta-llama/Llama-2-7b-hf --dataset wikitext2 --sparsity_ratio 0.5 --wandb_project_name pruning-llama2 --masks_per_iter 100 --nsamples 8 --save outdir --prune_frac 0.2 --bsz 1 --prune_method wanda
```


# Evaluations


```
prune_info='c4_masks-100'
location="/home/vashistt/Desktop/anlp-project/outdir_c4/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2_Adaptive=Yes"
outdir="/home/vashistt/anlp-project/finetuned_model/${prune_info}"
```
- ON AWS-- change the location and output_dir
    - Location: masks dir, Output: finetuned-model dir (wont be used for eval but add it for consistency)

```
location="/home/ec2-user/SageMaker/anlp-project/outdir_llama_2_7b/nsamp=8_sp=0.5_pfrac=0.2_bsz=1_ma_ratio=1.0_mpi=100_Lin.regtype=l1_pmethod=wanda_mlp_attn_ratio=1.0_Lin.regweight=100.0-0.0001-0_Lin.lr=100-10-1-0.1_Lin.bsz=32-64-128_Lin.nepochs=50_Lin.type=global_name=pruning-llama2-wikitext_Adaptive=Yes"

outdir="/home/ec2-user/SageMaker/anlp-project/finetuned_model"
```


### Evaluation Command after specifying location and outdir 

```
CUDA_VISIBLE_DEVICES=0 python3 lora_ft/Run_evals.py  --model_name_or_path "meta-llama/Llama-2-7b-hf"         --config_name "meta-llama/Llama-2-7b-hf"        --num_train_epochs 1         --block_size 512        --lora_r 128    --learning_rate 1e-4            --lora_alpha_ratio 4    --per_device_train_batch_size 1         --per_device_eval_batch_size 8       --do_train      --do_eval       --max_train_samples 15000       --max_eval_samples 128  --overwrite_output_dir  --output_dir "${outdir}"    --prune_info_path "${location}"     --hidden_mse_weight 0.0         --kl_weight 0.01        --dataset_name "wikitext"
```


- Generally these are the arguments 

```
CUDA_VISIBLE_DEVICES=0 python3 Run_evals.py \
	--model_name_or_path "meta-llama/Llama-2-7b-hf" \
	--config_name "meta-llama/Llama-2-7b-hf" \
	--num_train_epochs 1 \
	--block_size 512 \
	--lora_r 128 \
	--learning_rate 1e-4 \
	--lora_alpha_ratio 4 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 8 \
	--do_train  \
	--do_eval  \
 	--do_eleuther_eval True
	--max_train_samples 15000 \
	--max_eval_samples 128 \
	--overwrite_output_dir \
	--output_dir "${output_dir}" \
	--prune_info_path "${location}" \
	--hidden_mse_weight 0.0 \
	--kl_weight 0.01 \
	--dataset_name "wikitext"
```



---
## Prune Sheared LLaMA 2-7B
```
CUDA_VISIBLE_DEVICES=0 python3 main.py --model princeton-nlp/Sheared-LLaMA-2.7B --dataset wikitext2 --sparsity_ratio 0.5 --wandb_project_name pruning-sheared-llama2-wikitext --masks_per_iter 100 --nsamples 8 --save outdir --prune_frac 0.2 --bsz 1 --prune_method wanda
```
