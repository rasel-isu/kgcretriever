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
exp_name='w_can_type_test'
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
model_path = "models/LLaMA-HF/model/"+dir_name
os.makedirs(model_path, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model, cache_dir=model_path,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)
tokenizer_path = "models/LLaMA-HF/tokenizer/"+dir_name
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
     max_length=2048
    ):
    start_time = time()
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        temperature=temperature,
        num_return_sequences=1,
        eos_token_id=pipeline.tokenizer.eos_token_id,
        max_length=2048,
    )
    # max_new_tokens
    answer = f"{sequences[0]['generated_text'][len(prompt):]}\n"
    end_time = time()
    ttime = f"Total time: {round(end_time-start_time, 2)} sec."

    return  answer.strip() 


def add_to_json(file, new):
    with open(file) as f:
        old = json.load(f)
    old.append(new)
    with open(file, 'w') as f:
        json.dump(old, f, indent=2)


with open('DATASET/head_tail_query.json') as f:
    questions_and_desc = json.load(f)

# target = questions_and_desc['target_ent'] + '\n\n'
ent_above = 'of the above'
target = ''
prompt = f'{target}'+ """
Your task is predicting entity name. which entity will be the correct answer for the question below ?
What will be the type of predicted entity ? 
Enclose both predicted entity and type by [] bracket.
Question: {question}
Entity :
Type : 
"""
c_prd = 0
c_fld = 0
for c, i in enumerate(questions_and_desc['question']):
    # if c>10:
    #     break
    torch.cuda.empty_cache()
    try:
        prompt_text = prompt.format(question=f""" {i['input_text']}""")
        response = query_model(prompt_text, max_length=len(prompt_text)+32)
        i['prompt'] = prompt_text
        i['response'] = response
        print(response)
        add_to_json(f'REPORT/responses_{dir_name}_{exp_name}.json', i)
        c_prd+=1
        
    except Exception as e:
        i['error'] = str(e)
        add_to_json(f'REPORT/responses_{dir_name}_{exp_name}_failed.json', i)
        c_fld+=1

print(f'Predicted : {len(c_prd)}')
print(f'Failed : {len(c_fld)}')






