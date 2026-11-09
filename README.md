# Energy-Based Reasoners (EBR)

[Paper (arXiv)](https://arxiv.org/abs/2611.08977) | [Project page](https://energy-based-reasoners.github.io/)

> The repo is still under construction as we figure out ways to make energy-based training and eval faster as well as
> prepare ARC-AGI results. **ARC-AGI:** the dataset download and build scripts are included. ARC training configs and
> checkpoints will be released soon.

This is the official codebase for the paper ["Energy-Based Reasoners: Your Recurrent Reasoner Should be an
Energy-Based Model"](https://arxiv.org/abs/2611.08977).

EBR learns a scalar energy `E(z; x)` over a latent state `z` and reasons by gradient descent on it. EBR unifies evaluation, refinement and verification under one function.
During training, state-based explorative modeling (SXM) explores several latent trajectories and supervises the one with the best match. Forward
SXM picks among candidate states. Reverse SXM picks among the valid solutions of the input.

This repository trains and evaluates EBR on **Sudoku-Extreme, Maze-Hard, N-Queens (8x8, 10x10) and Graph Coloring
(8 and 10 vertices)**.

## Requirements

- Python 3.10 or newer
- CUDA 12.x and PyTorch 2.6 or newer (bf16 GPU recommended)

```bash
pip install --upgrade pip wheel setuptools
pip install torch --index-url https://download.pytorch.org/whl/cu124  # pick the build for your CUDA version
pip install -r requirements.txt
wandb login YOUR-LOGIN  # optional, to sync logs to Weights & Biases
```

## Dataset preparation

Download datasets and pretrained checkpoints:

```bash
bash scripts/download.sh              # datasets into data/, checkpoints into checkpoints/
bash scripts/download.sh datasets     # datasets only
bash scripts/download.sh checkpoints  # checkpoints only
```

Or build each dataset yourself:

```bash
bash scripts/build_all.sh   # builds the four benchmarks below (WITH_ARC=1 also builds ARC)

# Sudoku-Extreme (1,000 puzzles x 1,000 augmentations)
python -m dataset.build_sudoku_dataset --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Maze-Hard (30x30)
python -m dataset.build_maze_hard_dataset --output-dir data/maze-30x30-hard-1k

# N-Queens on GRAM's split (no download)
python -m dataset.build_nqueens_dataset --n 8 --output-dir data/nqueens-8
python -m dataset.build_nqueens_dataset --n 10 --output-dir data/nqueens-10

# Graph Coloring (GRAM authors' released data, pinned revision)
python -m dataset.build_graph_coloring_dataset --nodes 8 --output-dir data/graph-coloring-8
python -m dataset.build_graph_coloring_dataset --nodes 10 --output-dir data/graph-coloring-10

# ARC-AGI-1 / ARC-AGI-2 + ConceptARC (training release coming soon)
bash scripts/download_arc.sh
bash scripts/build_arc.sh
```

The Sudoku, Maze-Hard, Graph Coloring and ARC builders download their sources at pinned revisions.

## Experiments

`bash scripts/train.sh <task>` runs `python pretrain.py train/<task>` with the recipe in `config/train/<task>.yaml`:

```bash
bash scripts/train.sh sudoku             # Sudoku-Extreme,      60k steps, 1 H200, 3-4 h
bash scripts/train.sh maze_hard_fast     # Maze-Hard,           50k steps, 1 H200, 2-3 h
bash scripts/train.sh nqueens_8          # N-Queens 8x8,        30k steps, 1 H200, 3-4 h
bash scripts/train.sh nqueens_10         # N-Queens 10x10,      45k steps, 1 H200, 7-8 h
bash scripts/train.sh graph_coloring_8   # Graph Coloring, 8,   70k steps, 1 H200, 4-5 h
bash scripts/train.sh graph_coloring_10  # Graph Coloring, 10,  70k steps, 1 H200, 7-8 h
```

`sudoku` and `maze_hard_fast` use fused Triton kernels and need a CUDA GPU on Linux.

Runs use `seed=0` by default. Any config key can be overridden on the command line, e.g.
`bash scripts/train.sh sudoku seed=1 dataset.data_path=/data/sudoku`. `NPROC=<n>` runs on n GPUs. Runs write their
config, metrics and checkpoints (every 5000 steps) to `outputs/` (or `OUTPUT_ROOT`). Resume with
`load_checkpoint=<run dir>/checkpoints/step_<N>_<run_id>.pth`.

## Checkpoint paths

Pretrained checkpoints are on Hugging Face ([wxingxu/ebr](https://huggingface.co/wxingxu/ebr)).
`bash scripts/download.sh checkpoints` saves them under `checkpoints/`:

| Benchmark | Checkpoint | Evaluate |
|---|---|---|
| Sudoku-Extreme | `sudoku-extreme/sudoku.pth` | `bash scripts/eval.sh sudoku checkpoints/sudoku-extreme/sudoku.pth out/sudoku` |
| Maze-Hard | `maze-hard/maze.pth` | `bash scripts/eval.sh maze_hard checkpoints/maze-hard/maze.pth out/maze` |
| N-Queens 8x8 | `n-queens/nq8.pth` | `bash scripts/eval.sh nqueens_8 checkpoints/n-queens/nq8.pth out/nq8` |
| N-Queens 10x10 | `n-queens/nq10.pth` | `bash scripts/eval.sh nqueens_10 checkpoints/n-queens/nq10.pth out/nq10` |
| Graph Coloring, 8 | `graph-coloring/gc8.pth` | `bash scripts/eval.sh graph_coloring_8 checkpoints/graph-coloring/gc8.pth out/gc8` |
| Graph Coloring, 10 | `graph-coloring/gc10.pth` | `bash scripts/eval.sh graph_coloring_10 checkpoints/graph-coloring/gc10.pth out/gc10` |

Each file holds the model weights, their EMA and the training config.

## Evaluation

`bash scripts/eval.sh <task> <checkpoint> <out_dir>` runs `python evaluate.py --config config/eval/<task>.yaml` on a
checkpoint (EMA weights):

```bash
bash scripts/eval.sh sudoku           <checkpoint> out/sudoku  # all test boards, 8 samples, 64 segments
bash scripts/eval.sh maze_hard        <checkpoint> out/maze    # 1,000 mazes x 5 samples, 16 segments
bash scripts/eval.sh nqueens_8        <checkpoint> out/nq8     # GRAM's split, 20 samples per input (also: nqueens_10)
bash scripts/eval.sh graph_coloring_8 <checkpoint> out/gc8     # GRAM's split, 20 samples per input (also: graph_coloring_10)
```

- **N=1** is the first sample. **N>1** keeps the sample with the lowest final energy among the first N samples.

## Code layout

```
pretrain.py    training entry point (the training loop)
evaluate.py    evaluation entry point
config/        Hydra configs: cfg_pretrain.yaml, multi_solution.yaml (N-Queens / Graph Coloring base), arch/,
               dataset/, train/<task>.yaml, eval/<task>.yaml
scripts/       train.sh, eval.sh, download.sh (datasets and checkpoints), download / build datasets
dataset/       dataset builders, the metadata schema (common.py) and the data loader (puzzle_dataset.py)
models/        energy model and descent, attention, fused kernels, losses
evaluators/    in-training eval (in_loop.py), benchmark protocols (sudoku.py, maze.py, gram.py), scoring
utils/         config schema, optimizer, checkpointing, seeding and distributed helpers (runtime.py)
```

## Citation

```bibtex
@article{ebr2026,
  title={Energy-Based Reasoners: Your Recurrent Reasoner Should be an Energy-Based Model},
  author={William Xingxu Chen and Alexi Gladstone and Zeyu Michael Li and Heng Ji and Yilun Du},
  journal={arXiv preprint arXiv:2611.08977},
  year={2026}
}
```

This code builds on the Hierarchical Reasoning Model ([code](https://github.com/sapientinc/HRM)), the Tiny Recursive
Model ([code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)) and Generative Recursive Reasoning
([code](https://github.com/ahn-ml/gram-release)).
