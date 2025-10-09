# Test Code for Direct Quantized Training (DQT)
Code for ACML 2025 paper: Direct Quantized Training of Language Models with Stochastic Rounding (https://arxiv.org/abs/2412.04787).

We follow the code from https://huggingface.co/1bitLLM/bitnet_b1_58-large/tree/main. 

## Installation
Install the required libraries by `pip install -r requirements.txt`.


## Training
Prepare your huggingface access token and output directory in `train.py`.

In the code, we provide details of how we preprocess the wikipedia dataset. If you want to use larger datasets such as FineWeb, you can access it at https://huggingface.co/datasets/HuggingFaceFW/fineweb, and the preprocess procedure is consistent for both datasets.

Before training, please ensure that the bit width for stochastic rounding and initialization are the same. By default, we set them to 3 bits.

We use accelerator to do the multiple GPU training. Example execution command `CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --mixed_precision=fp16 train.py`
