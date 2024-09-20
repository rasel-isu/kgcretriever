# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import login
login(token='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')


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="models/LLaMA-HF")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir="models/LLaMA-HF")

print("done!")


