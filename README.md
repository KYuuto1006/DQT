# Test Code for Direct Quantized Training (DQT)
Install the required libraries by `pip install -r requirements.txt`.

Prepare your huggingface access token and output directory in `train.py`.

We use accelerator to do the multiple GPU training. Example execution command `CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --mixed_precision=fp16 train.py`
