import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langsmith import traceable
from src.schemas import RAGResponse

load_dotenv()

@traceable(name="get_chat_model")
def get_chat_model(temperature=0.2):
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["DEPLOYMENT_GPT_4O_MINI"],
        temperature=temperature,
    )

@traceable(name="get_gpt_40")
def get_gpt_40 (temperature=0.2):
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["DEPLOYMENT_GPT_4O_MINI"],
        temperature=temperature,
    )

@traceable(name="get_gpt_50")
def get_gpt_50 (temperature=0.2):
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["DEPLOYMENT_GPT_5"],
        temperature=temperature,
    )

@traceable(name="get_gpt_mini")
def get_gpt_mini (temperature=0.2):
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["DEPLOYMENT_GPT_4O_MINI"],
        temperature=temperature,
    )

@traceable(name="get_structured_chat_model")
def get_structured_chat_model():
    llm = get_chat_model()

    return llm.with_structured_output(
        RAGResponse
    )