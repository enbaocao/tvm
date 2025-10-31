# Megakernels!

High-performance LLM inference using instruction-based GPU kernels on NVIDIA Hopper (H100) and Blackwell (B200).

**⚡ Performance:** ~1000 tokens/second for Llama 3.2 1B on H100

## Quick Start

```bash
# 1. Clone and setup
git submodule update --init --recursive
python3 -m venv venv310
source venv310/bin/activate
pip install -e .

# 2. Login to HuggingFace (for Llama access)
pip install -U "huggingface_hub[cli]"
huggingface-cli login

# 3. Compile kernel
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)
export PYTHON_VERSION=3.10  # Match your python version
export GPU=H100
cd demos/low-latency-llama && make && cd ../..

# 4. Run
python megakernels/scripts/generate.py mode=mk prompt="Hello world" ntok=10
```

**See `QUICKSTART.md` for detailed setup instructions.**

## Low-Latency Llama Demo

First, to compile the megakernel, run:

```bash

# from the repo root
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)
export PYTHON_VERSION=3.12 # adjust if yours is different
export GPU=H100 # options are {H100, B200}, else defaults to B200
cd demos/low-latency-llama
make

```

To start an interactive chat session with the model, run:

```bash

# from the repo root
python megakernels/scripts/llama_repl.py

```

To benchmark the megakernel, run:

```bash

# from the repo root
python megakernels/scripts/generate.py mode=mk prompt="tell me a funny joke about cookies" ntok=100

```
