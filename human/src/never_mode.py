import os
import warnings
from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, message=".*FLAML.*")

load_dotenv()

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

agent_guess_animal = ConversableAgent(
    name="agent_guess_animal",
    system_message="""You are trying to guess an animal. Your behavior should be:
    - Ask one yes/no question at a time about the animal's characteristics
    - After getting 3-4 pieces of information, make a guess
    - If your guess is wrong, ask another question
    - Always make clear guesses like 'Is it a [animal]?'""",
    llm_config=llm_config,
    human_input_mode="NEVER",
)

# Start the conversation
agent_with_animal.initiate_chat(
    recipient=agent_guess_animal,
    message="I'm thinking of an animal. Ask me yes/no questions to guess what it is."
)
