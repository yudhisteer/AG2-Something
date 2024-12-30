import os
import warnings
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, message=".*FLAML.*")

load_dotenv()

"""
This script demonstrates a basic interaction between an AssistantAgent and UserProxyAgent using the AutoGen framework.

The AssistantAgent acts as an AI assistant that can engage in conversation and help with tasks.
The UserProxyAgent simulates a user, can execute code, and manages the interaction with the assistant.
"""

model = "gpt-3.5-turbo"
llm_config = {
    "model": model,
    "api_key": os.getenv("OPENAI_API_KEY"),
}

assistant = AssistantAgent(
    name="assistant",
    llm_config=llm_config,
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    llm_config=llm_config,
    code_execution_config={
        "work_dir": "code_execution",
        "use_docker": False,
    },
    human_input_mode="NEVER",
)


user_proxy.initiate_chat(
    recipient=assistant,
    message="Hello, how are you?",
)

