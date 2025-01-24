import subprocess
import os
from pyrit.memory.central_memory import CentralMemory
from pyrit.prompt_target import HuggingFaceChatTarget
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.common import initialize_pyrit, IN_MEMORY
import asyncio
import time

async def generate_from_prompts(model_id: str, prompt_list: list[str]):
    initialize_pyrit(IN_MEMORY)
    
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable not set")
    
    os.environ["HUGGINGFACE_TOKEN"] = hf_token
    
    model_times = {}

    print(f"Running model: {model_id}")

    try:
        # Initialize HuggingFaceChatTarget with the current model
        target = HuggingFaceChatTarget(model_id=model_id, use_cuda=False, tensor_format="pt", max_new_tokens=30)

        # Initialize the orchestrator
        orchestrator = PromptSendingOrchestrator(objective_target=target, verbose=False)

        # Record start time
        start_time = time.time()

        # Send prompts asynchronously
        responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore

        # Record end time
        end_time = time.time()

        # Calculate total and average response time
        total_time = end_time - start_time
        avg_time = total_time / len(prompt_list)
        model_times[model_id] = avg_time

        print(f"Average response time for {model_id}: {avg_time:.2f} seconds\n")

        # Print the conversations
        await orchestrator.print_conversations_async()  # type: ignore
        return avg_time
    
    except Exception as e:
        print(f"An error occurred with model {model_id}: {e}\n")
        model_times[model_id] = None
    

    # Print the model average time
    if model_times[model_id] is not None:
        print(f"{model_id}: {model_times[model_id]:.2f} seconds")
    else:
        print(f"{model_id}: Error occurred, no average time calculated.")

    memory = CentralMemory.get_memory_instance()
    memory.dispose_engine()

def run_garak(model_type, model_name):
    start_time = time.time()
    subprocess.run(["garak", "--model_type", model_type, "--model_name", model_name, "--probes", "test.TestOneProbe"], check=True)
    end_time = time.time()
    return end_time - start_time

def run_promptfoo(model_type, model_name, prompt_file):
    start_time = time.time()
    subprocess.run(["promptfoo", "eval", "--providers", f"{model_type}:text-generation:{model_name}", "--prompts", prompt_file], check=True)
    end_time = time.time()
    return end_time - start_time

# Define the models and prompts
models = [
            #"HuggingFaceTB/SmolLM-1.7B-Instruct", 
            #"HuggingFaceTB/SmolLM-135M-Instruct", 
            #"HuggingFaceTB/SmolLM-360M-Instruct",
            #"microsoft/Phi-3-mini-4k-instruct",
            #"Qwen/Qwen2-0.5B-Instruct",
            #"Qwen/Qwen2-1.5B-Instruct",
            "gpt2"
    ]

prompts = ["write a poem about the weather"]

async def main():
    # Iterate through models and prompts
    for model_name in models:
        for prompt in prompts:
            print(f"Testing model: {model_name} with prompt: {prompt}")

            time_pyrit = await generate_from_prompts(model_name, [prompt])
            print(f"Time taken by PyRit: {time_pyrit} seconds")
            #time_garak = run_garak("huggingface", model_name)
            #print(f"Time taken by Garak: {time_garak} seconds")
            #print("-" * 50)
            
            time_promptfoo = run_promptfoo("huggingface", model_name, "prompts.txt")
            print(f"Time taken by Promptfoo: {time_promptfoo} seconds")
            print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
