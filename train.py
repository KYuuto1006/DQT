from modeling_bitllama import BitnetForCausalLM
from tokenization_bitnet import BitnetTokenizer
from datasets import load_dataset, load_from_disk
import transformers
from transformers import Trainer, TrainingArguments, AutoTokenizer, LlamaConfig, DataCollatorForLanguageModeling, LlamaForCausalLM
import torch
import os
from accelerate import Accelerator
from custom_optimizer import MyAdamW
from utils_quant import weight_quant, weight_quant_n
import json

#device = "cuda" if torch.cuda.is_available() else "cpu"
accelerator = Accelerator(mixed_precision="fp16")
hf_access_token = "" # Add your Hugging Face access token here
tokenizer = BitnetTokenizer.from_pretrained("1bitLLM/bitnet_b1_58-large", token=hf_access_token) #meta-llama/Meta-Llama-3-8B


config = LlamaConfig(
    vocab_size=32002,
    hidden_size=768,
    intermediate_size=2048,
    num_hidden_layers=12,
    num_attention_heads=12,
    max_position_embeddings=512,
    rms_norm_eps=1e-5,
    initializer_range=0.02,
    use_cache=True,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    tie_word_embeddings=False,
    weight_bits=1,
    input_bits=8,
    torch_dtype=torch.float16,
)

model = BitnetForCausalLM(config=config)
model_size = sum(t.numel() for t in model.parameters())
print(f"model size: {model_size/1000**2:.1f}M parameters")

dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

train_test_split = dataset.train_test_split(test_size=0.01) 
train_dataset = train_test_split['train']
train_dataset = train_dataset.remove_columns(["title", "url", "id"])
dev_dataset = train_test_split['test']
dev_dataset = dev_dataset.remove_columns(["title", "url", "id"])
# train_dataset = train_dataset['text']
# dev_dataset = dev_dataset['text']

cache_dir = './cache'

# Check if the tokenized dataset exists in the cache
if os.path.exists(cache_dir):
    print("Loading tokenized dataset from cache...")
    tokenized_dataset = load_from_disk(os.path.join(cache_dir, 'train'))
    dev_dataset = load_from_disk(os.path.join(cache_dir, 'dev'))
    print(f"Chunked train dataset size: {len(tokenized_dataset)}")
    print(f"Chunked dev dataset size: {len(dev_dataset)}")
else:
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Dev dataset size: {len(dev_dataset)}")

    chunk_size = 512
    padding_token = tokenizer.pad_token_id
    def preprocess_function(examples):
        # Tokenize the text
        tokenized_inputs = tokenizer(examples['text'], truncation=False, padding=False)
        
        # Create chunks of the desired length
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]
        
        # Split into chunks
        chunked_input_ids = []  
        chunked_attention_mask = [] 
        for i in range(len(input_ids)):
            for j in range(0, len(input_ids[i]), chunk_size):
                chunk = input_ids[i][j:j + chunk_size]
                chunk_mask = attention_mask[i][j:j + chunk_size]
                # if len(chunk) < chunk_size:
                #     chunk += [padding_token] * (chunk_size - len(chunk))
                #     chunk_mask += [0] * (chunk_size - len(chunk_mask))
                chunked_input_ids.append(chunk)
                chunked_attention_mask.append(chunk_mask)
        
        return {"input_ids": chunked_input_ids, "attention_mask": chunked_attention_mask}
    tokenized_dataset = train_dataset.map(preprocess_function,
                                    batched=True,
                                    num_proc=32,
                                    remove_columns=["text"])

    dev_dataset = dev_dataset.map(preprocess_function,
                                    batched=True,
                                    num_proc=32,
                                    remove_columns=["text"])
    print(f"Chunked train dataset size: {len(tokenized_dataset)}")
    print(f"Chunked dev dataset size: {len(dev_dataset)}")
    
    os.makedirs(cache_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(os.path.join(cache_dir, 'train'))
    dev_dataset.save_to_disk(os.path.join(cache_dir, 'dev'))

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir='./results',
    logging_dir='./logs',
    evaluation_strategy="steps",
    gradient_accumulation_steps=1,
    eval_steps=0.2,
    save_steps=0.2,
    logging_steps=1,
    learning_rate=5e-4,
    weight_decay=0.01,
    num_train_epochs=1,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    seed=42,
    fp16=True,
)


for param in model.parameters():
    param.data = weight_quant_n(param.data, num_bits=3)

#optimizer = transformers.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
optimizer = MyAdamW(model.parameters(), weight_decay=training_args.weight_decay, lr=training_args.learning_rate)

#scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=160000)
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=55670)
#scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=2000)
#scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=55670, num_cycles=2)

trainer = Trainer(
    model=model,
    optimizers=(optimizer, scheduler),
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=dev_dataset,
    data_collator=data_collator,
)

model, trainer = accelerator.prepare(model, trainer)

trainer.train()