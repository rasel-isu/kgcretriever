from time import time
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from config import BASE_MODEL, EXPERIMENT_NAME
from utils import add_to_json

hf_token = "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"
login(token = hf_token)
base_model = BASE_MODEL
dir_name = base_model.replace('/', '_')

class Predictor:
    def __init__(self,pipeline):
        self.model = pipeline

    def predict_one(self, text):
        pass
    
    def predict_batch(self, texts):
        input_ids = self.tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True).to(self.device)


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
    ttime = round(end_time-start_time, 2)
    return  answer.strip() , ttime.strip()


class LLMPredictor(Predictor):
    def __init__(self):
        super().__init__(pipeline)

    def make_prompt(self, triple):
        with open('prompt.txt', 'r') as f:
            prompt = f.read().replace('___HEAD_ENTITY___', triple['head']).replace('___RELATION___', triple['relation'])
        return prompt

    def predict_dataset(self, dataset):
       # target = questions_and_desc['target_ent'] + '\n\n'
        ent_above = 'of the above'
        target = ''
        with open('prompt.txt', 'r') as f:
            prompt = f'{target}'+ f.read()
        c_prd = 0
        c_fld = 0
        for c, i in enumerate(dataset):
            # if c>10:
            #     break
            torch.cuda.empty_cache()
            try:
                prompt_text = self.make_prompt(i)
                response, time_taken = query_model(prompt_text, max_length=len(prompt_text)+32)
                i['time_taken'] = time_taken
                i['prompt'] = prompt_text
                i['response'] = response
                print(response)
                add_to_json(f'REPORT/responses_{dir_name}_{EXPERIMENT_NAME}.json', i)
                c_prd+=1
                
            except Exception as e:
                i['error'] = str(e)
                add_to_json(f'REPORT/responses_{dir_name}_{EXPERIMENT_NAME}_failed.json', i)
                c_fld+=1

        print(f'Predicted : {c_prd}')
        print(f'Failed : {c_fld}')

