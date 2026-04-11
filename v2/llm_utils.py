from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from enum import Enum


class Models(Enum):
    QWEN3_8B = 1
    QWEN3_32B = 2
    GEMINI = 3

class LLMUtils:
    def __init__(self, model, max_history=10):
        print("is Cuda available?: " + str(torch.cuda.is_available()))

        # Logging
        self.log_file = open("llm_log.txt", "a", buffering=1)  # line-buffered
        self.query_count = 0

        self.history = []
        self.max_history = max_history

        if model == Models.QWEN3_8B:
            model_name = "Qwen/Qwen3-8B"
        elif model == Models.QWEN3_32B:
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=cache_name
        )
        
    def query(self, prompt):
        self.history.append({"role": "user", "content": prompt})

        # Don't allow the history to get too large
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        text = self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # If we don't add the assistant keywords the model gets confused?
        self.history.append({"role": "assistant", "content": content})
        self.query_count += 1

        # ---- LOGGING ----
        self.log_file.write(f"\n=== QUERY {self.query_count} ===\n")
        self.log_file.write(f"PROMPT:\n{prompt}\n")
        self.log_file.write(f"RESPONSE:\n{content}\n")
        self.log_file.write("=" * 40 + "\n")
        self.log_file.flush()

        return content

if __name__ == "__main__":
    # This code only runs if this file is executed directly
    llm = LLMUtils(Models.QWEN3_32B)


    print("Is cuda available?: " + str(torch.cuda.is_available()))

    while True:
        prompt = input("Enter your prompt: ")
        response = llm.query(prompt)
        print(response)