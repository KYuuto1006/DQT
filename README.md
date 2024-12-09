# Test Code for Direct Quantized Training (DQT)
Code for paper: Direct Quantized Training of Language Models with Stochastic Rounding (https://arxiv.org/abs/2412.04787).

We follow the code from https://huggingface.co/1bitLLM/bitnet_b1_58-large/tree/main. 

## Installation
Install the required libraries by `pip install -r requirements.txt`.


## Training
Prepare your huggingface access token and output directory in `train.py`.

We use accelerator to do the multiple GPU training. Example execution command `CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --mixed_precision=fp16 train.py`
