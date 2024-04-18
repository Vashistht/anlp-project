## Sheared LLaMa

- iter 1 (p=.2 for .5 sparsity)

```
[v2]Iter :  95  PPL =  9.038905143737793
evaluating on wikitext2
nsamples 8
sample 0
[v1]Iter :  96  PPL =  10.772031784057617
evaluating on wikitext2
nsamples 8
sample 0
[v2]Iter :  96  PPL =  8.155013084411621
evaluating on wikitext2
nsamples 8
sample 0
[v1]Iter :  97  PPL =  14.738006591796875
evaluating on wikitext2
nsamples 8
sample 0
[v2]Iter :  97  PPL =  12.52479076385498
evaluating on wikitext2
nsamples 8
sample 0
[v1]Iter :  98  PPL =  23.333786010742188
evaluating on wikitext2
nsamples 8
sample 0
[v2]Iter :  98  PPL =  8.020745277404785
evaluating on wikitext2
nsamples 8
sample 0
[v1]Iter :  99  PPL =  11.160195350646973
evaluating on wikitext2
nsamples 8
sample 0
[v2]Iter :  99  PPL =  9.693778991699219
/home/ec2-user/anaconda3/envs/prune_llm/lib/python3.12/site-packages/plotly/matplotlylib/renderer.py:609: UserWarning:

I found a path object that I don't think is part of a bar chart. Ignoring.

Prune model
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 2560, padding_idx=0)
    (layers): ModuleList(
      (0): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (o_proj): Linear(in_features=2432, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5119, bias=False)
          (down_proj): Linear(in_features=5119, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5119, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (1): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2176, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2176, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2176, bias=False)
          (o_proj): Linear(in_features=2176, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5114, bias=False)
          (down_proj): Linear(in_features=5114, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5114, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (2): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5109, bias=False)
          (down_proj): Linear(in_features=5109, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5109, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (3): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5101, bias=False)
          (down_proj): Linear(in_features=5101, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5101, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (4): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5097, bias=False)
          (down_proj): Linear(in_features=5097, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5097, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (5): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (o_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5120, bias=False)
          (down_proj): Linear(in_features=5120, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5120, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (6): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (o_proj): Linear(in_features=2432, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5119, bias=False)
          (down_proj): Linear(in_features=5119, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5119, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (7): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2048, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2048, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5134, bias=False)
          (down_proj): Linear(in_features=5134, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5134, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (8): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (o_proj): Linear(in_features=2432, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5116, bias=False)
          (down_proj): Linear(in_features=5116, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5116, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (9): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (o_proj): Linear(in_features=2432, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5080, bias=False)
          (down_proj): Linear(in_features=5080, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5080, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (10): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (o_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5125, bias=False)
          (down_proj): Linear(in_features=5125, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5125, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (11): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5109, bias=False)
          (down_proj): Linear(in_features=5109, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5109, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (12): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (o_proj): Linear(in_features=2432, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5100, bias=False)
          (down_proj): Linear(in_features=5100, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5100, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (13): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5149, bias=False)
          (down_proj): Linear(in_features=5149, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5149, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (14): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (o_proj): Linear(in_features=2432, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5088, bias=False)
          (down_proj): Linear(in_features=5088, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5088, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (15): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (o_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5146, bias=False)
          (down_proj): Linear(in_features=5146, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5146, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (16): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (o_proj): Linear(in_features=2432, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5108, bias=False)
          (down_proj): Linear(in_features=5108, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5108, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (17): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (o_proj): Linear(in_features=2432, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5117, bias=False)
          (down_proj): Linear(in_features=5117, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5117, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (18): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (o_proj): Linear(in_features=2432, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5120, bias=False)
          (down_proj): Linear(in_features=5120, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5120, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (19): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5122, bias=False)
          (down_proj): Linear(in_features=5122, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5122, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (20): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5135, bias=False)
          (down_proj): Linear(in_features=5135, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5135, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (21): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (o_proj): Linear(in_features=2560, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5119, bias=False)
          (down_proj): Linear(in_features=5119, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5119, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (22): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (o_proj): Linear(in_features=2432, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5106, bias=False)
          (down_proj): Linear(in_features=5106, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5106, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (23): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5109, bias=False)
          (down_proj): Linear(in_features=5109, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5109, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (24): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2176, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2176, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2176, bias=False)
          (o_proj): Linear(in_features=2176, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5120, bias=False)
          (down_proj): Linear(in_features=5120, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5120, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (25): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2176, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2176, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2176, bias=False)
          (o_proj): Linear(in_features=2176, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5107, bias=False)
          (down_proj): Linear(in_features=5107, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5107, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (26): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5119, bias=False)
          (down_proj): Linear(in_features=5119, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5119, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (27): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5094, bias=False)
          (down_proj): Linear(in_features=5094, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5094, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (28): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5149, bias=False)
          (down_proj): Linear(in_features=5149, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5149, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (29): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2432, bias=False)
          (o_proj): Linear(in_features=2432, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5131, bias=False)
          (down_proj): Linear(in_features=5131, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5131, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (30): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5065, bias=False)
          (down_proj): Linear(in_features=5065, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5065, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
      (31): LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (k_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (v_proj): Linear(in_features=2560, out_features=2304, bias=False)
          (o_proj): Linear(in_features=2304, out_features=2560, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2560, out_features=5123, bias=False)
          (down_proj): Linear(in_features=5123, out_features=2560, bias=False)
          (up_proj): Linear(in_features=2560, out_features=5123, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=2560, out_features=32000, bias=False)
)
'''