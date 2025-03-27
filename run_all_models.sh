#!/bin/bash

# Model configurations
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "HuggingFaceH4/mistral-7b-grok"
    "microsoft/Phi-4-mini-instruct"
)

# Dataset configurations
DATASETS=(
    "keivalya/MedQuad-MedicalQnADataset,Question"
    "tatsu-lab/alpaca,instruction"
    "fka/awesome-chatgpt-prompts,prompt"
)

# # Validate models first; Uncomment this as needed. 
# echo "Validating models..."
# for model in "${MODELS[@]}"; do
#     python model_validation.py --model "$model"
#     if [ $? -ne 0 ]; then
#         echo "Model validation failed for $model"
#         exit 1
#     fi
# done

# Process each combination
for model in "${MODELS[@]}"; do
    MODEL_NAME=$(echo "$model" | cut -d'/' -f2)
    
    for dataset_config in "${DATASETS[@]}"; do
        # Split dataset config into name and column
        DATASET_NAME=$(echo "$dataset_config" | cut -d',' -f1)
        COLUMN_NAME=$(echo "$dataset_config" | cut -d',' -f2)
        DATASET_SHORT=$(echo "$DATASET_NAME" | cut -d'/' -f2)
        
        OUTPUT_BASE="results/${MODEL_NAME}_${DATASET_SHORT}"
        mkdir -p "results"
        
        echo "Processing $DATASET_NAME with $model"
        python local_only.py \
            --model "$model" \
            --input "$DATASET_NAME" \
            --is-hf-dataset \
            --text-column "$COLUMN_NAME" \
            --output "${OUTPUT_BASE}_sr_results.txt" \
            --raw-output "${OUTPUT_BASE}_sr_raw.txt" \
            --batch-size 256 \
            2>&1 | tee "${OUTPUT_BASE}_sr_log.txt"
    done
done