import os
import warnings
from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager
from dotenv import load_dotenv
import pprint

warnings.filterwarnings("ignore", category=UserWarning, message=".*FLAML.*")

load_dotenv()

"""
# ========
# This example illustrates the integration of a group chat within a sequential chat framework for effective travel planning.

# Agent Definitions:

# Flight_Agent, Hotel_Agent, Activity_Agent, Restaurant_Agent, Weather_Agent: Each agent specializes in a distinct aspect of travel planning.
# Travel_Planner_Agent: Responsible for summarizing the overall travel plan.
# Group Chat with Introductions:

# Facilitates the dialogue among travel planning agents, incorporating introductory messages.
# Group Chat Manager:

# Manages the group chat utilizing the defined LLM configuration.
# Sequential Chat:

# The sequential chat comprises the Travel_Planner_Agent and Group_Chat_Manager_With_Intros.
# The Group_Chat_Manager_With_Intros functions as a standard agent within the sequence, promoting collaborative travel planning.
# Process Overview
# The initial prompt "I'm planning a trip to Paris for the first week of September. Can you assist me in planning?" initiates the group chat.
# The Group_Chat_Manager_With_Intros directs the travel planning agents to create a detailed plan.
# The dialogue progresses through a series of chats, enhancing and summarizing the travel plan.
# This example showcases the integration of a group chat into a sequential chat to accomplish intricate, collaborative tasks.
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


# Group Chat in a Sequential Chat
# Group chat can also be used as a part of a sequential chat.
# In this case, the Group Chat Manager is treated as a regular agent in the sequence of two-agent chats.


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

# Create a Group Chat with introduction messages
group_chat_with_introductions = GroupChat(
    agents=[flight_agent, hotel_agent, activity_agent, restaurant_agent, weather_agent],
    messages=[],
    max_round=6,
    send_introductions=True,  # Send system messages to introduce each agent
)

# Create a Group Chat Manager
group_chat_manager_with_intros = GroupChatManager(
    groupchat=group_chat_with_introductions, 
    llm_config=llm_config
)

# Define a regular agent for the sequential chat
travel_planner_agent = ConversableAgent(
    name="Travel_Planner_Agent",
    system_message="You summarize the travel plan provided by the group chat.",
    llm_config=llm_config,
    description="Summarizes the travel plan.",
)

# Start a sequence of two-agent chats with the group chat manager as part of the sequence
chat_result = travel_planner_agent.initiate_chats(
    [
        {
            "recipient": group_chat_manager_with_intros,
            "message": "I'm planning a trip to Paris for the first week of September. Can you help me plan? I will be leaving from Miami and will stay for a week.",
            "summary_method": "reflection_with_llm",
        },
        {
            "recipient": group_chat_manager_with_intros,
            "message": "Please refine the plan with additional details.",
            "summary_method": "reflection_with_llm",
        },
    ]
)

# Print the output of each agent in the sequential chat
for result in chat_result:
    print(result.cost)

