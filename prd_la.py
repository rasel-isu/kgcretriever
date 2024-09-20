from time import time
import os
import json
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
hf_token = "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"
login(token = hf_token)
# base_model = "meta-llama/Meta-Llama-3-8B"
# base_model = "meta-llama/Meta-Llama-3-70B"
# base_model ='meta-llama/Meta-Llama-3.1-8B-Instruct'
exp_name='w_can_type'
# exp_name='with_candidate'
base_model ='meta-llama/Meta-Llama-3.1-70B-Instruct'
dir_name = base_model.replace('/', '_')
torch_dtype = torch.float16
attn_implementation = "eager"
# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
    
)
model_path = "kgllama/models/LLaMA-HF/model/"+dir_name
os.makedirs(model_path, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model, cache_dir=model_path,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)
tokenizer_path = "kgllama/models/LLaMA-HF/tokenizer/"+dir_name
os.makedirs(tokenizer_path, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(
    base_model, cache_dir=tokenizer_path)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)


def query_model(
    prompt, 
    temperature=0.7,
    max_length=512
    ):
    start_time = time()
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
    end_time = time()
    ttime = f"Total time: {round(end_time-start_time, 2)} sec."

    return  answer.strip() 



with open('questions_and_desc.json') as f:
    questions_and_desc = json.load(f)

target = questions_and_desc['target_ent'] + '\n\n'
ent_above = 'of the above'
# target = ''
prompt = f'{target}'+ """
Your task is to predict entity name. which entity will be the correct answer for the question below ?
What will be the type of entity ? 
Enclose both predicted entity and type by [] bracket.
Question: {question}
Answer:
"""

responses = []
for i in questions_and_desc['question']:
    response = query_model(
        prompt.format(question=f"""
                    {i['input_text']}
                    """),
        max_length=2048)
    i['response'] = response
    print(response)
    responses.append(i)

with open(f'responses_{dir_name}_{exp_name}.json', 'w') as f:
    json.dump(responses, f, indent=2)


