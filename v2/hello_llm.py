from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from enum import Enum


class Models(Enum):
    QWEN3 = 1
    GEMINI = 2

class LLMUtils:
    def __init__(self, model):
        if model == Models.QWEN3:
            model_name = "Qwen/Qwen3-8B"
            cache_name = "/scratch/general/vast/u1321655/"

            # Config that allows full model to be loaded into VRAM
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            # load the tokenizer and the model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_name)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                cache_dir=cache_name
            )
        
    def query(self, prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=16384
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        content = tokenizer.decode(output_ids, skip_special_tokens=True)

        return content

if __name__ == "__main__":
    # This code only runs if this file is executed directly
    llm = LLMUtils()
    llm.query()