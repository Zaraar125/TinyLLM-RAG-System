# app/llm.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "microsoft/phi-2"  # or try "google/gemma-2b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float32, 
    device_map="auto"
)

def generate_answer(context, query, max_new_tokens=150):
    prompt = f"""You are a helpful assistant. Use the context below to answer the user's question. Context: {context} Question: {query} Answer:"""
    # prompt = f"""You are a helpful assistant. Use the context below to answer the user's question. Question: {query} Answer:"""

    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)
