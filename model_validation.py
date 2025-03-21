import argparse
from vllm import LLM
import sys

def validate_model(model_id: str, test_prompt: str = "Is high blood pressure dangerous?") -> None:
    """Test model loading and basic inference. Exits if validation fails."""
    print(f"Validating model: {model_id}")
    try:
        max_seq_len = 256  # System prompt + examples + input + reminder + response
        max_seqs = 256     # Match our typical batch size
        # Actually load the model
        model = LLM(
            model=model_id,
            trust_remote_code=True,
            max_model_len=10000,  # Back to original working value
            gpu_memory_utilization=0.9,  # Known working value
            tensor_parallel_size=1,
            dtype="auto",
            enforce_eager=True
        )
        
        # Force model to do inference
        messages = [{"role": "user", "content": test_prompt}]
        result = model.chat(messages=messages)
        response = result[0].outputs[0].text
        
        print(f"Model loaded successfully. Test response: {response}")
        del model  # Explicitly cleanup
        
    except Exception as e:
        print(f"ERROR: Model validation failed for {model_id}")
        print(f"Error details: {str(e)}")
        sys.exit(1)  # Exit with error code

def main():
    parser = argparse.ArgumentParser(description='Validate LLM model loading and inference')
    parser.add_argument('--model', type=str, required=True, help='Model identifier to validate')
    args = parser.parse_args()
    
    validate_model(args.model)

if __name__ == "__main__":
    main()