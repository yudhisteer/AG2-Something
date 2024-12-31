import os
import warnings
from autogen import ConversableAgent, UserProxyAgent
from dotenv import load_dotenv
import pprint

warnings.filterwarnings("ignore", category=UserWarning, message=".*FLAML.*")

load_dotenv()

"""
- Implements a conversational AI system for travel planning.
- Utilizes the autogen library for AI agent management, specifically using the `ConversableAgent` class.
- The `ConversableAgent` class allows for the creation of agents that can engage in dialogue, manage conversation states, and execute tool calls.
- Features two agent instances: 
  - **Traveler Agent**: 
    - Interacts with users to gather information about their travel preferences.
    - Asks specific questions about destinations to tailor recommendations.
    - Expresses preferences and seeks itinerary advice based on user input.
  - **Guide Agent**: 
    - Provides customized recommendations based on the queries from the Traveler Agent.
    - Shares practical travel advice and tips to enhance the travel experience.
    - Suggests itineraries and activities that align with the user's interests.
- Initializes agents with specific configurations and system messages to define their roles and behaviors.
- Orchestrates a chat session between the Traveler Agent and Guide Agent to facilitate a dynamic dialogue.
- Focuses on travel options for a trip to Japan, ensuring relevant and engaging interactions.
- Enhances user travel planning through dynamic dialogue, allowing for a more personalized experience.
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

traveler_agent = ConversableAgent(
    name="Traveler_Agent",
    llm_config=llm_config,
    system_message="""You are a traveler planning a vacation. Your role is to:
1. Ask specific questions about destinations
2. Express your preferences and constraints
3. Seek advice on itinerary planning
Keep your messages clear and concise."""
)

guide_agent = ConversableAgent(
    name="Guide_Agent",
    llm_config=llm_config,
    system_message="""You are a travel guide helping travelers plan their trips. Your role is to:
1. Provide specific recommendations based on traveler requests
2. Share practical travel advice and tips
3. Suggest itineraries and activities
Keep responses focused and structured."""
)

try:
    chat_result = traveler_agent.initiate_chat(
        recipient=guide_agent,
        message="I'm interested in visiting Japan for 5 days. Can you suggest the main cities I should visit?",
        max_turns=2
    )
    
    print("**********************")
    print("Chat History:")
    print(chat_result.chat_history)
    print("**********************")
    print("Summary:")
    print(chat_result.summary)
    print("**********************")
    print("Default Summary Prompt:")
    print(ConversableAgent.DEFAULT_SUMMARY_PROMPT)
    print("**********************")
    print("Chat History:")
    pprint.pprint(chat_result.chat_history)
    print("**********************")
    
except Exception as e:
    print(f"Chat session failed: {str(e)}")