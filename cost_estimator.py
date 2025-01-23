import tiktoken

# GPT-4 pricing is charged PER 1000 TOKENS
# So if you have 2000 tokens, you need to divide by 1000 to get the multiplier
# for the per-1K pricing
GPT4_INPUT_COST_PER_1K = 0.03   # $0.03 per 1000 input tokens
GPT4_OUTPUT_COST_PER_1K = 0.06  # $0.06 per 1000 output tokens

def estimate_cost(prompt: str, num_runs: int, avg_output_tokens: int = 100) -> dict:
    """
    Estimate the cost of running garak tests with GPT-4.
    
    Args:
        prompt: The prompt text to be tested
        num_runs: Number of test runs
        avg_output_tokens: Estimated average tokens in response
    
    Returns:
        dict: Cost breakdown and total cost
        
    Note:
        OpenAI charges per 1000 tokens, so we divide the total tokens by 1000
        before multiplying by the cost per 1K tokens.
        Example: 2000 tokens = (2000/1000 = 2) * $0.03 = $0.06
    """
    input_tokens = count_tokens(prompt)
    
    total_input_tokens = input_tokens * num_runs
    total_output_tokens = avg_output_tokens * num_runs
    
    # Convert total tokens to thousands of tokens for pricing calculation
    input_cost = (total_input_tokens / 1000) * GPT4_INPUT_COST_PER_1K
    output_cost = (total_output_tokens / 1000) * GPT4_OUTPUT_COST_PER_1K
    
    total_cost = input_cost + output_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }

if __name__ == "__main__":
    # Example usage
    test_prompt = "Tell me about artificial intelligence"
    estimate = estimate_cost(test_prompt, num_runs=100)
    print(f"Cost estimate for 100 runs:")
    print(f"Input tokens: {estimate['input_tokens']}")
    print(f"Output tokens: {estimate['output_tokens']}")
    print(f"Input cost: ${estimate['input_cost']}")
    print(f"Output cost: ${estimate['output_cost']}")
    print(f"Total estimated cost: ${estimate['total_cost']}")
