import os
import warnings
from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager
from dotenv import load_dotenv
import pprint

warnings.filterwarnings("ignore", category=UserWarning, message=".*FLAML.*")

load_dotenv()

"""
- This script utilizes the autogen library to create a group chat framework.
- It defines multiple ConversableAgent instances, each responsible for a specific task related to travel planning.
- The agents include:
  - Flight_Agent: Provides flight options based on user input.
  - Hotel_Agent: Suggests hotels for the specified destination and dates.
  - Activity_Agent: Recommends activities and attractions at the destination.
  - Restaurant_Agent: Suggests dining options in the destination.
  - Weather_Agent: Provides weather forecasts for the travel dates.
- The agents are configured with a common language model (gpt-3.5-turbo) and specific system messages to guide their responses.
- The script sets up a collaborative interaction using the GroupChat and GroupChatManager classes, allowing users to engage with multiple agents simultaneously to plan their travel effectively.
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

flight_agent = ConversableAgent(
    name="Flight_Agent",
    system_message="You provide the best flight options for the given destination and dates.",
    llm_config=llm_config,
    description="Provides flight options.",
)

hotel_agent = ConversableAgent(
    name="Hotel_Agent",
    system_message="You suggest the best hotels for the given destination and dates.",
    llm_config=llm_config,
    description="Suggests hotel options.",
)

activity_agent = ConversableAgent(
    name="Activity_Agent",
    system_message="You recommend activities and attractions to visit at the destination.",
    llm_config=llm_config,
    description="Recommends activities and attractions.",
)

restaurant_agent = ConversableAgent(
    name="Restaurant_Agent",
    system_message="You suggest the best restaurants to dine at in the destination.",
    llm_config=llm_config,
    description="Recommends restaurants.",
)


weather_agent = ConversableAgent(
    name="Weather_Agent",
    system_message="You provide the weather forecast for the travel dates.",
    llm_config=llm_config,
    description="Provides weather forecast.",
)

# Create a Group Chat
group_chat = GroupChat(
    agents=[flight_agent, hotel_agent, activity_agent, restaurant_agent, weather_agent],
    messages=[],
    max_round=6,
)

# Create a Group Chat Manager
group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

# Initiate the chat with an initial message
chat_result = weather_agent.initiate_chat(
    group_chat_manager,
    message="I'm planning a trip to Paris for the first week of September. Can you help me plan? I will be departuring from Miami",
    summary_method="reflection_with_llm",
)