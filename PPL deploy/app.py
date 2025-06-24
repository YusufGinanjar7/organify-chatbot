from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

model_path = "D:\Kuliah\Kuliah\kuliah semester 6\PPL deploy\model\DialoGPT-medium"  # path ke file yg kamu download

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")

    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    response = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(response[:, inputs.shape[-1]:][0], skip_special_tokens=True)

    return {"response": reply}
