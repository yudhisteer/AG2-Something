import os
import warnings
from autogen import ConversableAgent, UserProxyAgent
from dotenv import load_dotenv
import pprint

warnings.filterwarnings("ignore", category=UserWarning, message=".*FLAML.*")

load_dotenv()

"""
# This script sets up a framework for sequential chat interactions using the autogen library.
# 
# - Defines the model to be used (gpt-3.5-turbo) and its configuration settings:
#   - Temperature for randomness in responses.
#   - API key fetched from environment variables.
#   - Configuration list for retries and timeout settings.
# 
# 
# - Initializes multiple ConversableAgent instances:
#   - Initial_Agent: Echoes back the input text.
#   - Uppercase_Agent: Converts input text to uppercase.
#   - WordCount_Agent: Counts the number of words in the input text.
#   - ReverseText_Agent: Reverses the input text.
# 
# - Each agent is configured with a specific system message and the same LLM configuration.
# 
# In this setup, messages are passed from one agent to the next in a chain-like manner. 
# Each agent processes the input it receives and produces an output that is then sent to the subsequent agent. 
# This allows for a collaborative interaction where each agent contributes its unique functionality to the overall conversation.




"""

model = "gpt-3.5-turbo"
llm_config = {
    "model": model,
    "temperature": 0.7,
    "api_key": os.getenv("OPENAI_API_KEY"),
    "config_list": [{
        "model": model,
        "api_key": os.getenv("OPENAI_API_KEY"),
        "timeout": 60,
        "max_retries": 3
    }]
}

if not llm_config["api_key"]:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# The Initial Agent always returns a given text.
initial_agent = ConversableAgent(
    name="Initial_Agent",
    system_message="You return me the text I give you.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# The Uppercase Agent converts the text to uppercase.
uppercase_agent = ConversableAgent(
    name="Uppercase_Agent",
    system_message="You convert the text I give you to uppercase.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# The Word Count Agent counts the number of words in the text.
word_count_agent = ConversableAgent(
    name="WordCount_Agent",
    system_message="You count the number of words in the text I give you.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)


# The Reverse Text Agent reverses the text.
reverse_text_agent = ConversableAgent(
    name="ReverseText_Agent",
    system_message="You reverse the text I give you.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# The Summarize Agent summarizes the text.
summarize_agent = ConversableAgent(
    name="Summarize_Agent",
    system_message="You summarize the text I give you.",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Start a sequence of two-agent chats.
# Each element in the list is a dictionary that specifies the arguments
# for the initiate_chat method.
chat_results = initial_agent.initiate_chats(
    [
        {
            "recipient": uppercase_agent,
            "message": "This is a sample text document.",
            "max_turns": 2,
            "summary_method": "last_msg",
        },
        {
            "recipient": word_count_agent,
            "message": "These are my numbers",
            "max_turns": 2,
            "summary_method": "last_msg",
        },
        {
            "recipient": reverse_text_agent,
            "message": "These are my numbers",
            "max_turns": 2,
            "summary_method": "last_msg",
        },
        {
            "recipient": summarize_agent,
            "message": "These are my numbers",
            "max_turns": 2,
            "summary_method": "last_msg",
        },
    ]
)

print("First Chat Summary: ", chat_results[0].summary)
print("Second Chat Summary: ", chat_results[1].summary)
print("Third Chat Summary: ", chat_results[2].summary)
print("Fourth Chat Summary: ", chat_results[3].summary)