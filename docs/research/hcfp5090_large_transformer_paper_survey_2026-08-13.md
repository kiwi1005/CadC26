# HCFP-5090 Large Transformer / Generative Model Survey

Date: 2026-08-13

## Bottom line

The primary-source evidence does **not** say that a giant general-purpose LLM is the standard answer for chip floorplanning / placement.

What the literature does support is:

1. **Transformer-style placement exists and is useful**, but the published chip-placement models are still mostly **moderate-sized** and task-specific.
2. **Diffusion / graph-transformer hybrids are the most plausible "bigger model" direction** for candidate generation.
3. **Specialized representation and training data matter more than raw scale** in the current placement literature.

For HCFP-5090, the best next experiment is **not** "replace everything with a giant LLM". The best evidence-backed experiments are:

- scale the existing sequence-policy head moderately;
- try a graph-transformer / diffusion candidate generator;
- keep the exact legalizer / repair tail unchanged.

## What I checked

Primary sources only:

- ChiPFormer paper and official repo
- Chip Placement with Diffusion Models paper and official repo
- AlphaChip / Circuit Training official repo
- TransPlace paper
- LayoutFormer++ paper
- Spatial Generalization for 3D Floorplanning paper

## Survey table

| Work | Type | What the source actually exposes | Why it matters for HCFP |
| --- | --- | --- | --- |
| ChiPFormer | Offline Decision Transformer for chip placement | Official repo config is `n_layer=6`, `n_head=8`, `n_embd=128`, `context_length=256`. The repo also says the method is two-stage: pretraining + fine-tuning, and uses 12 benchmarks for offline data. | This is the closest direct evidence that a transformer policy works for placement. It is not huge, but it is the most directly relevant transformer baseline. |
| Chip Placement with Diffusion Models | Diffusion / GNN-attention hybrid | Official repo has `train_large`, `train_medium`, `train_small`; `large.yaml` uses `hidden_size: 256`, `hidden_node_features: [256, 256, 256]`, `attention_node_features: [256, 256, 256]`. The repo also has a graph-transformer backbone with `num_heads: 4`, `model_dim: 128`, `num_layers: 4`. | This is the strongest "larger model" direction among placement papers. It is still a specialized graph/diffusion model, not a generic text LLM. |
| AlphaChip / Circuit Training | GCN-based RL with attention | Official repo says the model is `num_gcn_layers: 3`, `edge_fc_layers: 1`, `gcn_node_dim: 8`, plus an attention layer and a small value head. The repo also states it is a distributed deep-RL system for floorplans. | This is important as the canonical AlphaChip baseline, but it is **small**, not a giant transformer. It shows that strong placement results do not require LLM-scale models. |
| TransPlace | Graph neural network for global placement | Paper states TPGNN uses hidden dimensions `D_V = 64`, `D_U = 64`, `D_P = 8`, with `L = 3` message-passing layers and a two-stage inductive placement + fine-tuning pipeline. | Strong transfer on placement, but again the model is compact. This supports "better structure" over "bigger model". |
| LayoutFormer++ | Generic conditional layout generation | Transformer encoder-decoder with constraint serialization and decoding-space restriction. The paper does not expose a chip-placement-sized parameter count. | Useful as a structural analogy for tokenized constraint generation, but it is not a chip-placement paper. |
| Spatial Generalization for 3D Floorplanning | Decision Transformer / 3D floorplanning | The paper says `dmodel = 192`, with `4 attention layers` and `4 heads`. It explicitly argues that discrete canvas action spaces scale poorly and motivate continuous action spaces. | This is the closest transformer-style floorplanning result I found that explicitly talks about action-space scaling. It is still modest-sized, not giant-LLM-sized. |

## Source notes

### 1) ChiPFormer

Official paper:

- https://proceedings.mlr.press/v202/lai23c/lai23c.pdf

Key evidence:

- ChiPFormer is an offline RL / decision transformer method for chip placement.
- The official repo says pretraining and fine-tuning are separate stages.
- The official repo config uses `n_layer=6`, `n_head=8`, `n_embd=128`, `context_length=256`.

Why this matters:

- This is the strongest direct evidence that a transformer policy can work for placement.
- However, the published configuration is still modest; it is not an LLM-scale transformer.

Primary source refs:

- `laiyao1/chipformer@7906f3aa60202da1772992f1dc9da29436ac4051:run_dt_place.py:L109-L111`
- `laiyao1/chipformer@7906f3aa60202da1772992f1dc9da29436ac4051/README.md:L37-L54`

### 2) Chip Placement with Diffusion Models

Official paper:

- https://arxiv.org/pdf/2407.12282

Official repo:

- https://github.com/vint-1/chipdiffusion

Key evidence:

- The paper says the model is a diffusion model for simultaneous macro placement.
- The repo exposes `train_large`, `train_medium`, `train_small`.
- The `large` backbone config is `hidden_size: 256` with three 256-wide hidden-node and attention-node stacks.
- The repo also exposes a graph-transformer backbone with `num_heads: 4`, `model_dim: 128`, `num_layers: 4`.

Why this matters:

- If you want a larger generator for candidate placements, this is the most directly relevant primary source.
- It is still a graph / diffusion system, not a generic text LLM.

Primary source refs:

- `vint-1/chipdiffusion@6973e90ee1afc345b56da486bbf7dc9772ada1a4:README.md:L48-L66`
- `vint-1/chipdiffusion@6973e90ee1afc345b56da486bbf7dc9772ada1a4:diffusion/configs/model/size/large.yaml:L0-L2`
- `vint-1/chipdiffusion@6973e90ee1afc345b56da486bbf7dc9772ada1a4:diffusion/configs/model/graph-transformer.yaml:L0-L10`

### 3) AlphaChip / Circuit Training

Official repo:

- https://github.com/google-research/circuit_training

Key evidence:

- The repo describes AlphaChip as distributed deep reinforcement learning for chip floorplans.
- The model is compact: `num_gcn_layers: 3`, `edge_fc_layers: 1`, `gcn_node_dim: 8`.
- The Ariane config uses `max_grid_size: 128`, `sequence_length: 134`, `num_episodes_per_iteration: 1024`, `per_replica_batch_size: 128`, `num_epochs: 4`.
- The repo also notes a pre-trained checkpoint for 20 TPU blocks and recommends more in-distribution pretraining for best results.

Why this matters:

- AlphaChip is the canonical RL baseline, but it is not a giant transformer.
- It suggests that the win condition is usually representation + training regime, not raw model size.

Primary source refs:

- `google-research/circuit_training@c417a3a13f40867b649c719c03daaf1b39a909bc:README.md:L8-L12`
- `google-research/circuit_training@c417a3a13f40867b649c719c03daaf1b39a909bc:README.md:L49-L57`
- `google-research/circuit_training@c417a3a13f40867b649c719c03daaf1b39a909bc:README.md:L335-L341`
- `google-research/circuit_training@c417a3a13f40867b649c719c03daaf1b39a909bc:circuit_training/model/model_lib.py:L68-L75`
- `google-research/circuit_training@c417a3a13f40867b649c719c03daaf1b39a909bc:circuit_training/model/model_lib.py:L107-L167`
- `google-research/circuit_training@c417a3a13f40867b649c719c03daaf1b39a909bc:docs/ARIANE.md:L384-L397`

### 4) TransPlace

Official paper:

- https://arxiv.org/pdf/2501.05667

Key evidence:

- TPGNN uses `D_V = 64`, `D_U = 64`, `D_P = 8`.
- The model uses `L = 3` message-passing layers.
- The method is two-stage: inductive placement + circuit-adaptive fine-tuning.

Why this matters:

- This is a strong transferable placement model, but it is also compact.
- It supports the idea that graph structure and transfer matter more than brute-force scaling.

Primary source refs:

- `https://arxiv.org/pdf/2501.05667` lines `L116-L124`
- `https://arxiv.org/pdf/2501.05667` lines `L1338-L1345`

### 5) LayoutFormer++

Official paper:

- https://arxiv.org/pdf/2208.08037

Key evidence:

- It uses a Transformer encoder-decoder.
- It serializes constraints and uses decoding-space restriction.

Why this matters:

- It is a useful pattern for tokenized constraint handling.
- It is not chip placement, and the paper does not give a directly transferable chip-placement model size.

Primary source refs:

- `https://arxiv.org/pdf/2208.08037` lines `L13-L21`
- `https://arxiv.org/pdf/2208.08037` lines `L82-L100`

### 6) Spatial Generalization for 3D Floorplanning

Official paper:

- https://arxiv.org/pdf/2406.10538

Key evidence:

- The paper argues that discrete canvas action spaces scale poorly.
- It uses a decision transformer with `dmodel = 192`, `4 attention layers`, and `4 heads`.
- It explicitly frames floorplanning as a structured large discrete action space.

Why this matters:

- This is the clearest transformer-style floorplanning evidence I found for action-space scaling.
- It supports trying a transformer, but not necessarily a giant one.

Primary source refs:

- `https://arxiv.org/pdf/2406.10538` lines `L58-L61`
- `https://arxiv.org/pdf/2406.10538` lines `L383-L391`

## Concrete HCFP experiments

If you want to test "bigger model" quickly, I would do this order:

1. **ChiPFormer-style policy scaling**
   - Keep the current topology/candidate stack.
   - Try a larger sequence-policy head:
     - `n_embd: 128 -> 192 -> 256`
     - `n_layer: 6 -> 8`
     - `n_head: 8 -> 8/12`
   - Measure on `70, 89, 93, 98` first, then `large15`, then full100.

2. **ChipDiffusion-large candidate generator**
   - Use the `large` backbone (`hidden_size=256`) and the graph-transformer config (`4 layers`, `4 heads`, `model_dim=128`) as the first diffusion/transformer ablation.
   - Keep the exact legalizer and incumbent guard unchanged.

3. **TransPlace-style GNN control**
   - Use a compact graph model as a control condition.
   - This tells you whether the problem is really "more transformer" or just "better topology conditioning".

4. **No generic LLM until the above proves headroom**
   - I did not find primary-source evidence that a text-LLM-scale model is the standard winning axis for chip placement.
   - The published methods still rely on specialized action/state structure, not raw model size alone.

## Reusable takeaways

- The strongest directly relevant transformer placement paper is **ChiPFormer**, and its published config is modest, not giant.
- The strongest larger generative direction is **ChipDiffusion**, especially the `large` and `graph-transformer` configs.
- **AlphaChip and TransPlace are compact** and still strong, which is evidence against assuming that "bigger model" is the main missing ingredient.
- For HCFP-5090, the best research move is a **bounded scale-up experiment**, not a wholesale switch to an LLM.

## P9-T1 pilot result

The first implementation keeps every existing prediction head and the exact
tail, and replaces only the static scene encoder with dense pair-biased
self-attention.

| Model | Parameters | 1K four-case oracle | 5K four-case oracle | 5K large16 decode |
| --- | ---: | ---: | ---: | ---: |
| Message Tiny, 128x3 | 0.34M | 18.210 | 32.332 including decode failures | 14/16 |
| Transformer Medium, 192x4x6 | 2.28M | 19.350 | not extended | not run |
| Transformer Large, 256x6x8 | 5.61M | 18.878 | 17.600 | 15/16 |

The disjoint large16 audit shows the main early Large-model benefit is robust
topology construction: it reduced complete topology decode failures from two
cases to one.  On the successfully decoded subset, Tiny can still produce a
lower mean oracle, so this is a **MODIFY**, not a default promotion.  The next
experiment should use bucketed mini-batches and longer training before changing
the runtime default; current one-case training uses under 1 GB of the RTX 5090.

## Source URLs

- ChiPFormer paper: https://proceedings.mlr.press/v202/lai23c/lai23c.pdf
- ChiPFormer repo: https://github.com/laiyao1/chipformer
- Chip Placement with Diffusion Models paper: https://arxiv.org/pdf/2407.12282
- ChipDiffusion repo: https://github.com/vint-1/chipdiffusion
- AlphaChip / Circuit Training repo: https://github.com/google-research/circuit_training
- TransPlace paper: https://arxiv.org/pdf/2501.05667
- LayoutFormer++ paper: https://arxiv.org/pdf/2208.08037
- Spatial Generalization for 3D Floorplanning paper: https://arxiv.org/pdf/2406.10538
