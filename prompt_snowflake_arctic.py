import logging

from pyrit.common import IN_MEMORY, initialize_pyrit
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import HuggingFaceChatTarget
from pyrit.memory.central_memory import CentralMemory

import asyncio
import os

async def run_red_teaming_orchestrator(model_id: str):
    initialize_pyrit(IN_MEMORY)
    
    logging.basicConfig(level=logging.WARNING)

    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable not set")
    
    os.environ["HUGGINGFACE_TOKEN"] = hf_token
    
    model_times = {}

    print(f"Running model: {model_id}")

    try:
        # Initialize HuggingFaceChatTarget with the current model
        target = HuggingFaceChatTarget(model_id=model_id, use_cuda=False, tensor_format="pt", max_new_tokens=30)

        orchestrator = PromptSendingOrchestrator(objective_target=target, verbose=False)

        responses = await orchestrator.send_prompts_async(prompt_list=["say hello"])

        await orchestrator.print_conversations_async() 
    
    except Exception as e:
        print(f"An error occurred with model {model_id}: {e}\n")
        model_times[model_id] = None
    
    memory = CentralMemory.get_memory_instance()
    memory.dispose_engine()

async def main():
    await run_red_teaming_orchestrator("Snowflake/snowflake-arctic-embed-l-v2.0")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())