import os
import warnings
from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager, AssistantAgent
from dotenv import load_dotenv
import pprint

warnings.filterwarnings("ignore", category=UserWarning, message=".*FLAML.*")

load_dotenv()

"""
This script implements a customer support framework using multiple agents to handle various aspects of customer inquiries and responses. The key components include:

- **Inquiry_Agent**: Responsible for managing customer inquiries and classifying them based on their nature.
- **Response_Agent**: Provides automated responses tailored to the classification of the inquiries received.
- **Knowledge_Base_Agent**: Searches the company's knowledge base to find solutions to customer issues, ensuring accurate and relevant information is provided.
- **Troubleshooting_Agent**: Guides customers through troubleshooting steps to help resolve their issues effectively.
- **Feedback_Agent**: Collects customer feedback on the support experience to improve service quality and agent performance.

This structure allows for a streamlined and efficient customer support process, enhancing user satisfaction and operational effectiveness.
"""

model = "gpt-3.5-turbo"
llm_config = {
    "model": model,
    "temperature": 0.4,
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



# Define the customer inquiry agent
inquiry_agent = ConversableAgent(
    name="Inquiry_Agent",
    llm_config=llm_config,
    system_message="You handle customer inquiries and classify them.",
)

# Define the response agent
response_agent = ConversableAgent(
    name="Response_Agent",
    llm_config=llm_config,
    system_message="You provide automated responses based on the inquiry classification.",
)

# Define the knowledge base agent
knowledge_base_agent = ConversableAgent(
    name="Knowledge_Base_Agent",
    llm_config=llm_config,
    system_message="You search the company's knowledge base for solutions to customer issues.",
)

# Define the troubleshooting agent
troubleshooting_agent = ConversableAgent(
    name="Troubleshooting_Agent",
    llm_config=llm_config,
    system_message="You guide customers through troubleshooting steps to resolve their issues.",
)

# Define the feedback agent
feedback_agent = ConversableAgent(
    name="Feedback_Agent",
    llm_config=llm_config,
    system_message="You collect customer feedback on the resolution process.",
)

# Define the escalation agent
escalation_agent = ConversableAgent(
    name="Escalation_Agent",
    llm_config=llm_config,
    system_message="You identify cases that require human intervention.",
)

# Define the human support agent
human_support_agent = ConversableAgent(
    name="Human_Support_Agent",
    llm_config=llm_config,
    system_message="You connect customers with human support representatives.",
)

# Define the user proxy agent
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "my_code",
        "use_docker": False,
    },
)

# Register nested chats with the user proxy agent
user_proxy.register_nested_chats(
    [
        {
            "recipient": response_agent,
            "message": lambda recipient, messages, sender, config: f"Classify and respond to this inquiry: {messages[-1]['content']}",
            "summary_method": "last_msg",
            "max_turns": 1,
        },
        {
            "recipient": knowledge_base_agent,
            "message": lambda recipient, messages, sender, config: f"Search for solutions to this issue: {messages[-1]['content']}",
            "summary_method": "last_msg",
            "max_turns": 1,
        },
        {
            "recipient": troubleshooting_agent,
            "message": lambda recipient, messages, sender, config: f"Guide through troubleshooting for this issue: {messages[-1]['content']}",
            "summary_method": "last_msg",
            "max_turns": 1,
        },
        {
            "recipient": feedback_agent,
            "message": lambda recipient, messages, sender, config: f"Collect feedback on this resolution process: {messages[-1]['content']}",
            "summary_method": "last_msg",
            "max_turns": 1,
        },
        {
            "recipient": escalation_agent,
            "message": lambda recipient, messages, sender, config: f"Determine if this case needs human intervention: {messages[-1]['content']}",
            "summary_method": "last_msg",
            "max_turns": 1,
        },
    ],
    trigger=inquiry_agent,
)


# Define the initial customer inquiry
initial_inquiry = (
    """My internet is not working, and I have already tried rebooting the router."""
)

# Start the nested chat
user_proxy.initiate_chat(
    recipient=inquiry_agent,
    message=initial_inquiry,
    max_turns=2,
    summary_method="last_msg",
)