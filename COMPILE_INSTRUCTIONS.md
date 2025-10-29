# Compiling the Megakernel

## Quick Start (Copy-Paste)

```bash
# Set environment variables
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)
export PYTHON_VERSION=3.12  # Change to match your python3 version
export GPU=H100             # Options: 4090, A100, H100, or unset for B200

# Navigate to demo directory
cd demos/low-latency-llama

# Compile
make

# Return to root
cd ../..
```

## Step-by-Step Instructions

### 1. Check Prerequisites

```bash
# GPU check
nvidia-smi
# Should show: NVIDIA H100, A100, RTX 4090, or B200

# CUDA compiler check
nvcc --version
# Should show: CUDA 12.x or higher

# Python check
python3 --version
# Should show: Python 3.9+ (3.12 recommended)
```

### 2. Set Environment Variables

```bash
# From repository root (/Users/enbao/projects/tvm)
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)

# Match your Python version (check with: python3 --version)
export PYTHON_VERSION=3.9   # or 3.10, 3.11, 3.12, 3.13

# Match your GPU
export GPU=H100    # Options: 4090, A100, H100, (or unset for B200)
```

**GPU Selection Guide:**
- **RTX 4090**: `export GPU=4090` (compute capability 8.9)
- **A100**: `export GPU=A100` (compute capability 8.0)
- **H100**: `export GPU=H100` (compute capability 9.0a)
- **B200**: Don't set GPU variable, defaults to B200 (compute capability 10.0a)

### 3. Compile the Kernel

```bash
cd demos/low-latency-llama
make
```

**Expected output:**
```
nvcc llama.cu ... -o mk_llama.cpython-312-darwin.so
```

The compilation will create a shared library: `mk_llama.cpython-3XX-<platform>.so`

**Compilation time:** ~2-5 minutes on first build

### 4. Verify Compilation

```bash
ls -lh mk_llama*.so
# Should show the compiled shared library
```

### 5. Return to Root

```bash
cd ../..
```

## Troubleshooting

### Error: "nvcc: command not found"

**Solution:** Install CUDA Toolkit
```bash
# Check if CUDA is installed
ls /usr/local/cuda

# If not, download from: https://developer.nvidia.com/cuda-downloads
# Or use conda:
conda install -c nvidia cuda-toolkit
```

### Error: "Python.h: No such file or directory"

**Solution:** Install Python development headers
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# CentOS/RHEL
sudo yum install python3-devel

# Or use conda environment (recommended)
conda install python=3.12
```

### Error: "pybind11/pybind11.h: No such file or directory"

**Solution:** Install pybind11
```bash
pip install pybind11
```

### Error: "Cannot find -lpython3.12"

**Solution:** Verify PYTHON_VERSION matches your installation
```bash
# Check available Python versions
ls /usr/lib/libpython* 2>/dev/null || ls /usr/local/lib/libpython*

# Or check with pkg-config
pkg-config --list-all | grep python

# Update PYTHON_VERSION to match
export PYTHON_VERSION=3.11  # or whatever you have
```

### Error: Architecture mismatch (wrong GPU selected)

**Symptoms:**
```
ptxas error: unsupported gpu architecture
```

**Solution:** Set correct GPU variable
```bash
# Check your GPU
nvidia-smi | grep "NVIDIA"

# Set matching GPU variable
export GPU=A100  # or 4090, H100, etc.
```

### Compilation is slow or hangs

**Cause:** NVCC compiling for multiple architectures

**Solution:** Already optimized - Makefile only compiles for your selected GPU
- H100: `-arch=sm_90a`
- B200: `-arch=sm_100a`
- A100: `-arch=sm_80`
- 4090: `-arch=sm_89`

### Error: "ThunderKittens not found"

**Solution:** Initialize git submodules
```bash
git submodule update --init --recursive
```

## Testing After Compilation

### Quick Test - Generate Tokens

```bash
python3 megakernels/scripts/generate.py mode=mk prompt="Hello world" ntok=10
```

**Expected output:**
```
Loading model...
Compiling kernel...
Generating 10 tokens...
Output: Hello world, this is a test of the megakernel...
Time: ~X ms/token
```

### Interactive REPL

```bash
python3 megakernels/scripts/llama_repl.py
```

### Run Diff Test

```bash
python3 megakernels/scripts/diff_test.py
```

**This compares megakernel output vs PyTorch reference**

## Clean Build

If you need to recompile from scratch:

```bash
cd demos/low-latency-llama
make clean
make
cd ../..
```

## Makefile Details

The Makefile in `demos/low-latency-llama/Makefile`:
- Uses `nvcc` to compile CUDA code
- Links against Python, pybind11, CUDA libraries
- Creates a Python extension module
- Optimizes for your specific GPU architecture
- Includes ThunderKittens headers
- Compiles with `-O3` optimization and fast math

Key compiler flags:
```makefile
NVCCFLAGS=-O3 -std=c++20 -DNDEBUG
NVCCFLAGS+=-I${THUNDERKITTENS_ROOT}/include
NVCCFLAGS+=$(shell python3 -m pybind11 --includes)
NVCCFLAGS+=-arch=sm_90a  # For H100
```

## Multi-GPU Systems

If you have multiple GPUs:

```bash
# See all GPUs
nvidia-smi

# Set specific GPU for testing
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
python3 megakernels/scripts/generate.py
```

## Docker Option (Alternative)

If you have CUDA driver but want isolated environment:

```bash
# Use NVIDIA PyTorch container
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:24.09-py3 \
  bash

# Inside container
cd /workspace
export THUNDERKITTENS_ROOT=/workspace/ThunderKittens
export MEGAKERNELS_ROOT=/workspace
export PYTHON_VERSION=3.10
export GPU=H100
cd demos/low-latency-llama
make
```

## Remote Compilation

If compiling on a remote server:

```bash
# SSH to GPU server
ssh user@gpu-server

# Clone and compile
git clone <repo-url>
cd <repo>
git submodule update --init --recursive
pip install -e .

# Follow compilation steps above
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
# ... etc
```

## Next Steps After Successful Compilation

1. **Test inference:**
   ```bash
   python3 megakernels/scripts/generate.py mode=mk
   ```

2. **Benchmark performance:**
   ```bash
   python3 megakernels/scripts/generate.py mode=mk ntok=100
   ```

3. **Compare with PyTorch:**
   ```bash
   python3 megakernels/scripts/diff_test.py
   ```

4. **Interactive mode:**
   ```bash
   python3 megakernels/scripts/llama_repl.py
   ```
