import os
import torch
from fastapi import FastAPI, Request, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DialoGPT Chat API", version="1.0.0")

# Global variables untuk model dan tokenizer
tokenizer = None
model = None

def load_model():
    """Load model dan tokenizer dengan error handling"""
    global tokenizer, model
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/DialoGPT-medium",
            cache_dir="./model_cache"
        )
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/DialoGPT-medium",
            cache_dir="./model_cache",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Set pad_token jika belum ada
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Load model saat aplikasi startup"""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "DialoGPT Chat API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.post("/chat")
async def chat(request: Request):
    """Chat endpoint dengan error handling yang lebih baik"""
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        data = await request.json()
        user_input = data.get("message", "").strip()
        
        if not user_input:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Encode input
        inputs = tokenizer.encode(
            user_input + tokenizer.eos_token, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Generate response dengan parameter yang lebih baik
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[-1] + 50,  # Batasi panjang response
                min_length=inputs.shape[-1] + 5,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2
            )
        
        # Decode response
        response = tokenizer.decode(
            outputs[:, inputs.shape[-1]:][0], 
            skip_special_tokens=True
        ).strip()
        
        return {
            "response": response if response else "I'm not sure how to respond to that.",
            "input_length": inputs.shape[-1],
            "output_length": outputs.shape[-1] - inputs.shape[-1]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
