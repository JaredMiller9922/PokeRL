from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from enum import Enum
import re
import time

class Models(Enum):
    QWEN3_8B = 1
    QWEN3_32B = 2
    GEMINI = 3

app = FastAPI()

tokenizer = None
model = None
model_name = None
# Dictionary with {session_id: message_list[]}
enable_history = False
histories = {}

def load_model(model_id: int):
    global tokenizer, model, model_name

    if model_id == 1:
        model_name = "Qwen/Qwen3-8B"
    elif model_id == 2:
        model_name = "Qwen/Qwen3-32B"

    cache_name = "/scratch/general/vast/u1321655/"

    # Config that allows full model to be loaded into VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_name
    )
    model.eval()

class QueryRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    session_id: str = "default"
    thinking: bool = False


@app.post("/query")
def query_model(req: QueryRequest):
    global histories
    if req.session_id not in histories:
        histories[req.session_id] = []

    if enable_history:
        history = histories[req.session_id]
        history.append({"role": "user", "content": req.prompt})

        text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=req.thinking,
        )
    else:
        messages = [{"role": "user", "content": req.prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=req.thinking,
        )
    # print(f"PROMPT LENGTH: {len(text)}", flush=True)

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    start = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,
    )
    gen_time = time.time() - start
    print(f"GEN TIME: {gen_time:.2f}s", flush=True)

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)

    if enable_history:
        history.append({"role": "assistant", "content": content})

        max_history = 2
        if len(history) > max_history:
            history = history[-max_history:]
            histories[req.session_id] = history

    # extract last float in the output
    match = re.findall(r"[-+]?\d*\.\d+|\d+", content)

    if match:
        clean_output = match[-1]  # take last number
    else:
        clean_output = "0.0"     # fallback

    # print("RAW:", content, flush=True)
    # print("PARSED:", clean_output, flush=True)

    return {"response": clean_output}
