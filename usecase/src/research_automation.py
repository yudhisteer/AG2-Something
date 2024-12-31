import os
import warnings
from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager, AssistantAgent
from dotenv import load_dotenv
import pprint
import pandas as pd
import autogen

warnings.filterwarnings("ignore", category=UserWarning, message=".*FLAML.*")

load_dotenv()


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

def read_article(file_path):
    with open(file_path, "r") as file:
        return file.read()


# Create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
)

# Create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "code_execution",
        "use_docker": False,
    },
)

# Define the agents
content_analysis_agent = AssistantAgent(
    name="Content_Analysis_Agent",
    llm_config=llm_config,
    system_message="""
    You analyze the submitted article for structure, coherence, and completeness.
    """,
)

# Task 1: Find research papers
task1 = """
Find arxiv papers that discuss the applications of machine learning in healthcare.
"""
user_proxy.initiate_chat(assistant, message=task1)

# Task 2: Analyze the results to list the specific healthcare applications
task2 = "Analyze the results to list the specific healthcare applications studied by these papers."
user_proxy.initiate_chat(assistant, message=task2, clear_history=False)

# Task 3: Generate a bar chart
task3 = """Use this data to generate a bar chart of healthcare applications and the number of papers in each application and save it to a file.
"""
user_proxy.initiate_chat(assistant, message=task3, clear_history=False)


file_path = r"usecase\data\article.txt"
article_content = read_article(file_path)

initial_task = f"Analyze the following article for structure, coherence, and completeness: {article_content}"

content_result = user_proxy.initiate_chat(
    recipient=content_analysis_agent,
    message=initial_task,
    max_turns=2,
    summary_method="last_msg",
)

style_review_agent = AssistantAgent(
    name="Style_Review_Agent",
    llm_config=llm_config,
    system_message="""
    You review the article for language use, tone, and style consistency.
    """,
)

fact_checking_agent = AssistantAgent(
    name="Fact_Checking_Agent",
    llm_config=llm_config,
    system_message="""
    You verify the factual accuracy of the content.
    """,
)

editorial_feedback_agent = AssistantAgent(
    name="Editorial_Feedback_Agent",
    llm_config=llm_config,
    system_message="""
    You provide comprehensive feedback and suggestions for improvement.
    """,
)

final_review_agent = AssistantAgent(
    name="Final_Review_Agent",
    llm_config=llm_config,
    system_message="""
    You summarize the overall quality of the article and readiness for publication.
    """,
)




style_result = user_proxy.initiate_chat(
    recipient=style_review_agent,
    message=content_result,
    max_turns=2,
    summary_method="last_msg",
)

fact_result = user_proxy.initiate_chat(
    recipient=fact_checking_agent,
    message=style_result,
    max_turns=2,
    summary_method="last_msg",
)

feedback_result = user_proxy.initiate_chat(
    recipient=editorial_feedback_agent,
    message=fact_result,
    max_turns=2,
    summary_method="last_msg",
)


final_summary = user_proxy.initiate_chat(
    recipient=final_review_agent,
    message=feedback_result,
    max_turns=2,
    summary_method="last_msg",
)

print("Final Summary or Report:")
print(final_summary)