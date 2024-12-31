import os
import warnings
from autogen import ConversableAgent, UserProxyAgent
from dotenv import load_dotenv
from typing import Annotated as A

warnings.filterwarnings("ignore", category=UserWarning, message=".*FLAML.*")

load_dotenv()



model = "gpt-3.5-turbo"
llm_config = {
    "model": model,
    "temperature": 0,
    "api_key": os.getenv("OPENAI_API_KEY"),
}

def add_numbers(a: A[int, "The first number to add"], b: A[int, "The second number to add"]) -> str:
    return f"The sum of {a} and {b} is {a + b}"

def multiply_numbers(a: A[int, "The first number to multiply"], b: A[int, "The second number to multiply"]) -> str:
    return f"The product of {a} and {b} is {a * b}" 

assistant = ConversableAgent(
    name="CalculatorAssistant",
    system_message="You are a helpful AI calculator. Return 'TERMINATE' when the task is done.",
    llm_config=llm_config,
)


user_proxy = ConversableAgent(
    name="User",
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    llm_config=llm_config,
    human_input_mode="NEVER"
)


# register tools with the assistant
assistant.register_for_llm(name="add_numbers", description="Add two numbers")(add_numbers)
assistant.register_for_llm(name="multiply_numbers", description="Multiply two numbers")(multiply_numbers)

# register tools with the user_proxy
user_proxy.register_for_execution(name="add_numbers")(add_numbers)
user_proxy.register_for_execution(name="multiply_numbers")(multiply_numbers)


user_proxy.initiate_chat(
    recipient=assistant,
    message="Add 3 and 2"
)


