import os
import warnings
from autogen import ConversableAgent
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, message=".*FLAML.*")

load_dotenv()

"""
This script demonstrates a basic implementation of a ConversableAgent using the AutoGen framework.

The ConversableAgent is configured with GPT-3.5-turbo model and can generate responses to messages.
It is set up with code execution disabled and no human input mode, making it a simple conversational agent.

The script sends a test message to the agent and prints its response to demonstrate basic functionality.

"""

model = "gpt-3.5-turbo"
llm_config = {
    "model": model,
    "api_key": os.getenv("OPENAI_API_KEY"),
}

agent = ConversableAgent(
    name="simple_agent",
    llm_config=llm_config,
    code_execution_config=False,
    human_input_mode=False
)

response = agent.generate_reply(messages=[{"role": "user", "content": "Hello, how are you?"}])
print(response)
