# ComplianceChatOOD
A repository focused specifically on the testing of various methods for enforcing extreme constraints on LLM chat responses using methods external to the typical LLM-user interaction.


## Acquiring the general container


## Launching the container
[at the moment, the modified triton container is only locally available]

The call should look roughly like this:

docker run --gpus all --rm -it \
  --shm-size=2G \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /home/repo/.cache/huggingface:/root/.cache/huggingface \
  -v /home/repo/model_repository_vllm:/models \
  -e HF_HOME=/root/.cache/huggingface \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  f1e3a6801bd5 \
  tritonserver --model-repository=/models \
  --log-verbose=1 \
  --strict-model-config=false

This launches a triton server with an updated vllm installation that's required to use the Microsoft Phi4 model without hitches in the rope scaling.
