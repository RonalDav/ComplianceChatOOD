# ComplianceChatOOD
A repository focused specifically on the testing of various methods for enforcing extreme constraints on LLM chat responses using methods external to the typical LLM-user interaction.


## Primary operation

Right now, the three files should be quite portable to other environments. The bash script is the best way to ensure you will run all the models with every dataset, but on a local 24GB VRAM card this can still take up to 8 hours, though, it should be about half that in the worst case.



## To visitors:

Welcome!
These are a handful of scripts that we made for the purpose of our submitted work to PEARC25.

This is intended to demonstrate an approach where one might need to enforce hard constraints on the type of user inputs that are considered acceptable use for your live application. As a demonstration, we simply show that this approach is simple, lightweight, and effective which positions it as a highly valuable deterring mechanism for providing an empty response to a user query/prompt that is beyond the scope of what one's application should respond to. 

From this, there are many followup techniques that could provide further value to a high-end, user-facing LLM application that runs at a high cost:

- finetune on the output patterns for a robust, aligned filtering mechanism that suits your purpose
- modify the system prompt, memory adjustments, and reminder to enforce your own rules
- greatly expand the historical examples to avoid some needs for finetuning with low risk of rejecting acceptable inputs
- modify it into a natural language routing mechanism
...

There are many options for building upon this foundation in order to provide more efficient, constrained serving for your more sensitive applications.


(figures to come later, contact for citation information if it is not posted here yet)


<!-- ## Acquiring the general container


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

This launches a triton server with an updated vllm installation that's required to use the Microsoft Phi4 model without hitches in the rope scaling. -->
