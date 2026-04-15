import uvicorn
import llm_server
import argparse

parser = argparse.ArgumentParser(description="A script that greets users.")

parser.add_argument("-m", "--model", type=int, help = "1. Qwen3-8B, 2.Qwen3-32B")
args = parser.parse_args()

# Set the model before starting the server
if (args.model == 1):
    print("Using Qwen3-8B")
elif (args.model == 2):
    print("Using Qwen3-32B")


llm_server.load_model(model_id=args.model)   # 1 = Qwen3-8B, 2 = Qwen3-32B

if __name__ == "__main__":
    uvicorn.run(llm_server.app, host="127.0.0.1", port=8000)