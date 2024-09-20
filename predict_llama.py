import json
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import time
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset
from trl import SFTTrainer, setup_chat_format

from huggingface_hub import login
#from kaggle_secrets import UserSecretsClient
#user_secrets = UserSecretsClient()

hf_token = "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"

login(token = hf_token)

base_model = "meta-llama/Meta-Llama-3-8B"

torch_dtype = torch.float16
attn_implementation = "eager"
# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)
# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model, cache_dir="kgllama/models/LLaMA-HF/model",
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model, cache_dir="kgllama/models/LLaMA-HF/tokenizer")

pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="cuda",
)

def query_model(
    prompt,
    temperature=0.7,
    max_length=512
    ):
    start_time = time.time()
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        temperature=temperature,
        num_return_sequences=1,
        eos_token_id=pipeline.tokenizer.eos_token_id,
        max_length=max_length,
    )
    answer = f"{sequences[0]['generated_text'][len(prompt):]}\n"
    end_time = time.time()
    ttime = f"Total time: {round(end_time-start_time, 2)} sec."

    return prompt + " " + answer  + " " +  ttime



with open('questions_and_desc.json') as f:
    questions_and_desc = json.load(f)

target = questions_and_desc['target_ent']

prompt = f'{target}\n\n'+ """
Your task is to predict which of the above entity will be the correct answer for the question below.
Question: {question}
Answer:
Reason:
"""

responses = []
for i in tqdm(questions_and_desc['question']):
    response = query_model(
        prompt.format(question=f"""
                    {i['input_text']}
                    """),
        max_length=512)
    i['response'] = response
    print(response)
    responses.append(i)

with open('responses.json', 'w') as f:
    json.dump(responses, f, indent=2)

