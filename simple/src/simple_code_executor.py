import os
import warnings
from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, message=".*FLAML.*")

load_dotenv()

model = "gpt-3.5-turbo"
llm_config = {
    "model": model,
    "api_key": os.getenv("OPENAI_API_KEY"),
}

assistant = AssistantAgent(
    name="Assistant",
    llm_config=llm_config,
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    llm_config=llm_config,
    code_execution_config={
        "work_dir": "code_execution",
        "use_docker": False,
    },
    human_input_mode="ALWAYS",
)

user_proxy.initiate_chat(
    recipient=assistant,
    message="Plot a chart of META and TESLA stock prices from 2022 to 2024",
)
