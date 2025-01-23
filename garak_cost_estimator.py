import tiktoken
from pathlib import Path
import importlib
import inspect
from garak.probes.base import Probe
from garak.generators.test import Blank
from typing import List, Dict
import pkgutil
import garak.probes

# GPT-4 pricing per 1K tokens
GPT4_INPUT_COST_PER_1K = 0.03
GPT4_OUTPUT_COST_PER_1K = 0.06

def get_probe_prompts(probe_names: List[str]) -> Dict[str, List[str]]:
    """Get all prompts from specified probes."""
    prompts = {}
    for probe_name in probe_names:
        try:
            module_path, class_name = probe_name.rsplit('.', 1)
            module = importlib.import_module(f"garak.probes.{module_path}")
            probe_class = getattr(module, class_name)
            
            # Initialize probe and generator
            test_generator = Blank()
            probe = probe_class()
            probe.generator = test_generator
            
            # Get prompts - try different methods depending on probe type
            prompts[probe_name] = []
            
            # Try accessing prompts directly if available
            if hasattr(probe, 'prompts'):
                prompts[probe_name].extend(probe.prompts)
            
            # Try using _generate() if available
            if hasattr(probe, '_generate'):
                try:
                    for attempt in probe._generate():
                        if attempt and hasattr(attempt, 'prompt'):
                            prompts[probe_name].append(attempt.prompt)
                except Exception as e:
                    print(f"Warning: Error during _generate for {probe_name}: {e}")
            
            # If no prompts found, try the generate() method
            if not prompts[probe_name] and hasattr(probe, 'generate'):
                try:
                    for attempt in probe.generate():
                        if attempt and hasattr(attempt, 'prompt'):
                            prompts[probe_name].append(attempt.prompt)
                except Exception as e:
                    print(f"Warning: Error during generate for {probe_name}: {e}")
            
            # Deduplicate prompts
            prompts[probe_name] = list(set(prompts[probe_name]))
            
        except Exception as e:
            print(f"Error loading probe {probe_name}: {e}")
            continue
    return prompts

def estimate_garak_cost(probe_names: List[str], 
                       generations_per_prompt: int = 10,
                       avg_response_tokens: int = 100) -> dict:
    """
    Estimate GPT-4 costs for running garak with specific probes.
    
    Args:
        probe_names: List of probe names to estimate (e.g., ['dan.Dan_11_0'])
        generations_per_prompt: Number of generations per prompt (default 10)
        avg_response_tokens: Average expected response length in tokens
    
    Returns:
        dict: Detailed cost breakdown per probe and total
    """
    encoder = tiktoken.encoding_for_model("gpt-4")
    results = {
        'probes': {},
        'total': {
            'input_tokens': 0,
            'output_tokens': 0,
            'input_cost': 0,
            'output_cost': 0,
            'total_cost': 0
        }
    }
    
    # Get prompts for each probe
    probe_prompts = get_probe_prompts(probe_names)
    
    for probe_name, prompts in probe_prompts.items():
        total_input_tokens = 0
        
        # Calculate tokens for each prompt
        for prompt in prompts:
            input_tokens = len(encoder.encode(prompt))
            total_input_tokens += input_tokens * generations_per_prompt
        
        total_output_tokens = len(prompts) * generations_per_prompt * avg_response_tokens
        
        input_cost = (total_input_tokens / 1000) * GPT4_INPUT_COST_PER_1K
        output_cost = (total_output_tokens / 1000) * GPT4_OUTPUT_COST_PER_1K
        
        results['probes'][probe_name] = {
            'num_prompts': len(prompts),
            'total_generations': len(prompts) * generations_per_prompt,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'input_cost': round(input_cost, 2),
            'output_cost': round(output_cost, 2),
            'total_cost': round(input_cost + output_cost, 2)
        }
        
        # Update totals
        results['total']['input_tokens'] += total_input_tokens
        results['total']['output_tokens'] += total_output_tokens
        results['total']['input_cost'] += input_cost
        results['total']['output_cost'] += output_cost
        results['total']['total_cost'] += input_cost + output_cost
    
    # Round total costs
    results['total']['input_cost'] = round(results['total']['input_cost'], 2)
    results['total']['output_cost'] = round(results['total']['output_cost'], 2)
    results['total']['total_cost'] = round(results['total']['total_cost'], 2)
    
    return results

def list_available_probes() -> List[str]:
    """Get a list of all available probe names in garak."""
    probes = []
    for _, module_name, _ in pkgutil.iter_modules(garak.probes.__path__):
        try:
            module = importlib.import_module(f"garak.probes.{module_name}")
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, Probe) and obj != Probe:
                    probes.append(f"{module_name}.{name}")
        except Exception as e:
            print(f"Warning: Could not load module {module_name}: {e}")
    return probes

if __name__ == "__main__":
    # List available probes
    available_probes = list_available_probes()
    print("\nAvailable probes:")
    for probe in available_probes:
        print(f"  {probe}")
    
    estimate = estimate_garak_cost(available_probes)
    
    print("\nCost Estimate per Probe:")
    for probe, data in estimate['probes'].items():
        print(f"\n{probe}:")
        print(f"  Prompts: {data['num_prompts']}")
        print(f"  Total generations: {data['total_generations']}")
        print(f"  Estimated cost: ${data['total_cost']}")
    
    print(f"\nTotal estimated cost: ${estimate['total']['total_cost']}")
