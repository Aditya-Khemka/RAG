import os
from dotenv import load_dotenv
from importlib.metadata import version

from langchain_openai import AzureChatOpenAI

load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():

    llm1 = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["DEPLOYMENT_GPT_4O_MINI"],
        temperature=0,
    )
    response1 = llm1.invoke("Say 'setup complete!' in one word")
    print(f"response1: {response1.content}")


    llm2 = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["DEPLOYMENT_GPT_4O_MINI"],
        temperature=0,
    )
    response2 = llm2.invoke("Say 'setup complete!' in two words")
    print(f"response2: {response2.content}")

    print("Setup complete!")


if __name__ == "__main__":
    main()