import io
import csv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Body
from fastapi.param_functions import Form
from pydantic import BaseModel
import torch
import torch.nn as nn
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextGenerator:
    def __init__(self, model_name="gpt2", device=None):
        """
        Initialize the text generator with a pre-trained model.
        Popular model options:
        - "gpt2" (small, fast)
        - "gpt2-medium" (medium size)
        - "EleutherAI/gpt-neo-125M" (Neo alternative)
        - "facebook/opt-350m" (Meta's OPT model)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Loading model {model_name} on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.to(self.device)
        
        print("Model loaded successfully!")

    def generate_text(
        self,
        prompt,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        no_repeat_ngram_size=2
    ):
        """
        Generate text based on a prompt.
        
        Args:
            prompt (str): The input prompt
            max_length (int): Maximum length of generated text (including prompt)
            num_return_sequences (int): Number of different sequences to generate
            temperature (float): Controls randomness (higher = more random)
            top_k (int): Number of highest probability tokens to consider
            top_p (float): Cumulative probability threshold for token selection
            do_sample (bool): If True, sample from distribution; if False, use greedy decoding
            no_repeat_ngram_size (int): Size of n-grams that shouldn't be repeated
        """
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode and return the generated text
        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return generated_texts
    
model = TextGenerator("gpt2") # lightweight, easier to run locally

class TextInput(BaseModel):
    text: str

def process_text(text):
    results = model.generate_text(text)
    return results[0]

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/process")
async def process_input(
    text_input: TextInput = Body(None), 
):
    if text_input:
        result = process_text(text_input.text)
        return {"result": result}
    else:
        raise HTTPException(status_code=400, detail="No input provided")

@app.post("/process_csv")
async def process_csv(file: UploadFile = File(...)):
    if file.content_type != "text/csv":
        raise HTTPException(status_code=400, detail="Invalid file type")
    content = await file.read()
    csv_reader = csv.reader(io.StringIO(content.decode('utf-8')))
    results = []
    for row in csv_reader:
        result = process_text(row[0])
        results.append(result) 
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
