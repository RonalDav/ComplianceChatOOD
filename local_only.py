import argparse
from pathlib import Path
from typing import List, Dict, Optional
import json

# Separate critical imports
from vllm import LLM, SamplingParams
from datasets import load_dataset

def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run medical/HPC domain filtering with LLM')
    parser.add_argument('--input', type=str, help='Input file with prompts or HuggingFace dataset name')
    parser.add_argument('--output', type=str, default='results.txt', help='Output file for responses')
    parser.add_argument('--raw-output', type=str, default='raw_results.txt', help='Output file for raw conversation history')
    parser.add_argument('--is-hf-dataset', action='store_true', help='Input is a HuggingFace dataset')
    parser.add_argument('--dataset-split', type=str, default='train', help='Dataset split to use')
    parser.add_argument('--text-column', type=str, default='text', help='Column name for text in dataset')
    parser.add_argument('--batch-size', type=int, default=128, 
                       help='Batch size for processing prompts')
    parser.add_argument('--model', type=str, default="microsoft/Phi-4-mini-instruct",
                       help='Model identifier to use')
    return parser

def initialize_model(model_id: str) -> LLM:
    """Initialize LLM with specified model and optimal memory settings."""
    try:
        model = LLM(
            model=model_id,
            trust_remote_code=True,
            max_model_len=10000,  # Back to original working value
            gpu_memory_utilization=0.9,  # Known working value
            tensor_parallel_size=1,
            dtype="auto",
            enforce_eager=True
        )
        return model
    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        print("Ensure you have the correct model weights downloaded and CUDA is available")
        raise

def get_default_sampling_params() -> SamplingParams:
    return SamplingParams(
        max_tokens=1500,
        temperature=0.0,
    )

def get_system_prompt() -> str:
    return """You are a specialized AI filtering assistant focused on healthcare, medical topics, and high-performance computing.
For every user prompt, you must decide if the input is a medical question.
Prepend your response with an indicator using double angle brackets:
    <<YES>> if the input is a medical question, or <<NO>> if it is not.
If the input is not medical, only respond with <<NO>> I can only assist with healthcare and HPC related questions.
If the input is medical, simply reply <<YES>>.
Ensure that the indicator appears at the very beginning of your response with no extra tokens.
Keep your deliberation and response extremely brief or only use <<YES>> and <<NO>> if posisble.
Your memory shows you have done this succesfully before."""

def get_reminder_prompt() -> str:
    return "\nRemember: Only respond with <<YES>> if this is a medical question, or <<NO>> if not. No additional text."

def initialize_conversation() -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": "Is high blood pressure dangerous?"},
        {"role": "assistant", "content": "<<YES>>"}, 
        {"role": "user", "content": "what is the primary color of the un flag?"}, 
        {"role": "assistant", "content": "<<NO>>"},
        {"role": "user", "content": "What is a leiomeiosarcoma and how is it supposed to be spelled?"},
        {"role": "assistant", "content": "<<YES>>"}
    ]
    return messages

def load_prompts_from_file(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_prompts_from_dataset(dataset_name: str, split: str, text_column: str) -> List[str]:
    dataset = load_dataset(dataset_name, split=split)
    return [item[text_column] for item in dataset]

def chat_with_model(llm: LLM, messages: List[Dict[str, str]], 
                   sampling_params: SamplingParams) -> str:
    output = llm.chat(messages=messages, sampling_params=sampling_params)
    return output[0].outputs[0].text

def chat_with_model_batch(llm: LLM, message_batches: List[List[Dict[str, str]]], 
                         sampling_params: SamplingParams) -> List[str]:
    # Process multiple conversations in parallel
    outputs = llm.chat(messages=message_batches, sampling_params=sampling_params)
    return [output.outputs[0].text for output in outputs]

def process_prompts(llm: LLM, prompts: List[str], 
                   output_file: str, raw_output_file: str,
                   batch_size: int):  # Remove default value here
    sampling_params = get_default_sampling_params()
    reminder = get_reminder_prompt()
    
    # Initialize counters
    counts = {"YES": 0, "NO": 0, "undetermined": 0}
    undetermined_responses = []
    results = []
    
    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        message_batches = []
        
        # Prepare batch of messages
        for prompt in batch:
            messages = initialize_conversation()
            augmented_prompt = f"{prompt}{reminder}"
            messages.append({"role": "user", "content": augmented_prompt})
            message_batches.append(messages)
        
        # Get responses for entire batch
        responses = chat_with_model_batch(llm, message_batches, sampling_params)
        
        # Process batch responses
        for j, (prompt, response) in enumerate(zip(batch, responses)):
            results.append((prompt, response))
            
            # Update counters
            if response.strip() == "<<YES>>":
                counts["YES"] += 1
            elif response.strip() == "<<NO>>":
                counts["NO"] += 1
            else:
                counts["undetermined"] += 1
                undetermined_responses.append((prompt, response))
        
        # Progress update for batches
        if (i + batch_size) % (batch_size * 10) == 0:
            print(f"Processed {i + len(batch)}/{len(prompts)} prompts...")
            print(f"Current counts: {counts}")
    
    # Write results in bulk
    with open(output_file, 'w') as f_out, open(raw_output_file, 'w') as f_raw:
        for idx, (prompt, response) in enumerate(results, 1):
            messages = initialize_conversation()
            messages.append({"role": "user", "content": f"{prompt}{reminder}"})
            messages.append({"role": "assistant", "content": response})
            
            f_out.write(f"Prompt {idx}: {prompt}\n")
            f_out.write(f"Response: {response}\n\n")
            f_raw.write(json.dumps(messages) + "\n")
    
    # Write categorized results
    write_categorized_results(results, output_file, counts, reminder)
    
    # Final statistics
    total = sum(counts.values())
    print("\nFinal Statistics:")
    print(f"Total prompts processed: {total}")
    print(f"YES responses: {counts['YES']} ({(counts['YES']/total)*100:.2f}%)")
    print(f"NO responses: {counts['NO']} ({(counts['NO']/total)*100:.2f}%)")
    print(f"Undetermined: {counts['undetermined']} ({(counts['undetermined']/total)*100:.2f}%)")

def write_categorized_results(results: List[tuple], 
                            base_output_file: str,
                            counts: Dict[str, int],
                            reminder: str):
    # Get base filename without extension
    base_name = Path(base_output_file).stem
    base_dir = Path(base_output_file).parent
    
    # Prepare category files
    categories = {
        "yes": [(p, r) for p, r in results if r.strip() == "<<YES>>"],
        "no": [(p, r) for p, r in results if r.strip() == "<<NO>>"],
        "undetermined": [(p, r) for p, r in results if r.strip() not in ["<<YES>>", "<<NO>>"]]
    }
    
    # Write to category-specific files
    for category, items in categories.items():
        output_file = base_dir / f"{base_name}_{category}.txt"
        with open(output_file, 'w') as f:
            f.write(f"# {category.upper()} responses ({len(items)} items)\n\n")
            for idx, (prompt, response) in enumerate(items, 1):
                f.write(f"Prompt {idx}: {prompt}\n")
                f.write(f"Response: {response}\n\n")

def main():
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Initialize model with specified model ID
    llm = initialize_model(args.model)
    
    if args.is_hf_dataset:
        prompts = load_prompts_from_dataset(args.input, args.dataset_split, args.text_column)
    else:
        prompts = load_prompts_from_file(args.input)
    
    # Pass batch_size from args
    process_prompts(llm, prompts, args.output, args.raw_output, args.batch_size)
    print(f"Processing complete. Results written to {args.output} and {args.raw_output}")

if __name__ == "__main__":
    main()