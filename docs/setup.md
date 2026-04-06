# Setup Guide

Step-by-step instructions for setting up the inference engine from a fresh machine.

---

## 1. CUDA and driver requirements

| Component | Minimum | Recommended |
|---|---|---|
| NVIDIA driver | 530 | 545+ |
| CUDA toolkit | 12.1 | 12.4 |
| GPU architecture | Turing (RTX 20xx, T4) | Ampere (A100, RTX 30xx, A10G) or later |
| Python | 3.10 | 3.11 |

### Verify your setup

```bash
# Check driver version
nvidia-smi

# Check CUDA toolkit
nvcc --version

# Check GPU memory
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
```

### Install NVIDIA driver (Ubuntu 22.04)

```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install driver only (CUDA toolkit comes via pip with torch)
sudo apt-get install -y nvidia-driver-545
sudo reboot

# After reboot, verify
nvidia-smi
```

---

## 2. Python environment

```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` pulls PyTorch with the CUDA 12.1 index. If you have a different CUDA version, change the index URL at the top of the file:

```
# CUDA 11.8
--extra-index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4
--extra-index-url https://download.pytorch.org/whl/cu124
```

### Verify PyTorch sees your GPU

```python
import torch
print(torch.cuda.is_available())        # True
print(torch.cuda.get_device_name(0))    # NVIDIA A10G
print(torch.cuda.get_device_properties(0).total_memory / 1e9)  # 24.0
```

---

## 3. vLLM installation notes

vLLM ships pre-built CUDA kernels and takes a few minutes to install. If installation fails:

```bash
# Common fix: ensure pip and setuptools are current
pip install --upgrade pip setuptools wheel

# Then retry
pip install vllm==0.4.3
```

On machines with CUDA 12.4+, use the latest vLLM:

```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124
```

---

## 4. bitsandbytes installation notes

bitsandbytes requires a working CUDA installation and compiles a small native extension on first use:

```bash
pip install bitsandbytes

# Verify
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

If you see `CUDA Setup failed`, check that `nvcc` is on your PATH:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

## 5. Running the offline quantization job

Quantization needs a GPU and takes 30–60 minutes. Run it on a machine with enough VRAM to load the full FP16 model (14 GB for 7B, 26 GB for 13B).

```bash
# AWQ — 7B model, needs ~16 GB VRAM for the calibration pass
python scripts/quantize.py awq \
    --model Qwen/Qwen2-7B-Instruct \
    --output ./artifacts/qwen2-7b-awq

# GPTQ — same VRAM requirement
python scripts/quantize.py gptq \
    --model meta-llama/Llama-2-13b-hf \
    --output ./artifacts/llama2-13b-gptq \
    --bits 4 --group-size 128
```

For gated models (Llama, Mistral), set your HuggingFace token:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
huggingface-cli login --token $HF_TOKEN
```