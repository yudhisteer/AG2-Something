import os
import warnings
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, message=".*FLAML.*")

load_dotenv()

"""
This script demonstrates the ALWAYS human input mode in AutoGen agents.
It creates a guessing game where:
- One agent thinks of an elephant and can only answer yes/no questions
- A human proxy agent makes guesses through human input
- Every interaction requires human input (no autonomous responses)
- The game ends automatically if 'elephant' is guessed correctly

The ALWAYS mode ensures human input is required for every interaction.
This shows how agents can be used to create interactive experiences
where a human user maintains full control of the conversation flow.
The conversation will continue prompting for human input until either:
1. The human chooses to exit
2. The correct animal (elephant) is guessed

"""

model = "gpt-3.5-turbo"
llm_config = {
    "model": model,
    "api_key": os.getenv("OPENAI_API_KEY"),
}


agent_with_animal = ConversableAgent(
    name="agent_with_animal",
    system_message="""You are thinking of an elephant. When asked questions:
    - Answer only with 'yes' or 'no'
    - You can add ONE short hint after your yes/no if relevant
    - Never reveal directly that it's an elephant
    - Only confirm if someone explicitly guesses 'elephant'""",
    llm_config=llm_config,
    is_termination_msg=lambda msg: "elephant" in msg["content"].lower(),
    human_input_mode="NEVER",
)

human_proxy = UserProxyAgent(
    name="human_proxy",
    llm_config=False, # no llm for human
    human_input_mode="ALWAYS", # human input is always required
    code_execution_config=False, # no code execution for human
)


# Start the conversation
result = human_proxy.initiate_chat(
    recipient=agent_with_animal,
    message="Parrot"
)
