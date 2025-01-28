import subprocess
import os
from pyrit.memory.central_memory import CentralMemory
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.common import initialize_pyrit, IN_MEMORY
import asyncio
import time
from pyrit_exp.gemini_target import GeminiTarget
from tabulate import tabulate
from dataclasses import dataclass
from typing import Optional

@dataclass
class TestResult:
    model: str
    prompt: str
    pyrit_time: Optional[float] = None
    garak_time: Optional[float] = None
    promptfoo_time: Optional[float] = None

async def generate_from_prompts(model_id: str, prompt_list: list[str]):
    initialize_pyrit(IN_MEMORY)
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    model_times = {}

    print(f"Running model: {model_id}")

    try:
        # Initialize HuggingFaceChatTarget with the current model
        target = GeminiTarget(endpoint=f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={google_api_key}")

        # Initialize the orchestrator
        orchestrator = PromptSendingOrchestrator(objective_target=target, verbose=True, scorers=None)

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

def run_garak():
    start_time = time.time()
    subprocess.run(["garak", "--model_type", "rest", "-G", "gemini_rest_garak.json", "--probes", "test.TestOneProbe"], check=True)
    end_time = time.time()
    return end_time - start_time

def run_promptfoo(model_name, prompt_file):
    start_time = time.time()
    subprocess.run(["promptfoo", "eval", "--providers", f"google:{model_name}", "--prompts", prompt_file], check=True)
    end_time = time.time()
    return end_time - start_time

# Define the models and prompts
models = [
            "gemini-2.0-flash-thinking-exp"
    ]

prompts = ["write a poem about the weather"]

async def main():
    results = []
    
    # Iterate through models and prompts
    for model_name in models:
        for prompt in prompts:
            print(f"Testing model: {model_name} with prompt: {prompt}")
            
            result = TestResult(model=model_name, prompt=prompt)
            
            try:
                result.pyrit_time = await generate_from_prompts(model_name, [prompt])
                print(f"Time taken by PyRit: {result.pyrit_time} seconds")
                print("-" * 50)
            except Exception as e:
                print(f"Error in PyRit: {e}")
                
            try:
                result.garak_time = run_garak()
                print(f"Time taken by Garak: {result.garak_time} seconds")
                print("-" * 50)
            except Exception as e:
                print(f"Error in Garak: {e}")
            
            try:
                result.promptfoo_time = run_promptfoo(model_name, "prompts.txt")
                print(f"Time taken by Promptfoo: {result.promptfoo_time} seconds")
                print("-" * 50)
            except Exception as e:
                print(f"Error in Promptfoo: {e}")
                
            results.append(result)
    
    # Create table
    headers = ["Model", "Prompt", "PyRit Time (s)", "Garak Time (s)", "Promptfoo Time (s)"]
    table_data = [
        [
            r.model,
            r.prompt[:30] + "..." if len(r.prompt) > 30 else r.prompt,
            f"{r.pyrit_time:.2f}" if r.pyrit_time else "Error",
            f"{r.garak_time:.2f}" if r.garak_time else "Error",
            f"{r.promptfoo_time:.2f}" if r.promptfoo_time else "Error"
        ]
        for r in results
    ]
    
    # Print table
    print("\nResults Summary:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Save results to file
    with open("benchmark_results.txt", "w") as f:
        f.write(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    asyncio.run(main())
