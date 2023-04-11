import openai
from config import Config
from colorama import Fore, Style

cfg = Config()

openai.api_key = cfg.openai_api_key

# Overly simple abstraction until we create something better
def create_chat_completion(messages, model=None, temperature=None, max_tokens=None)->str:
    """Create a chat completion using the OpenAI API"""
    if cfg.use_azure:
        openai.api_type = "azure"
        # openai.api_version = "2023-03-15-preview" 
        openai.api_version = cfg.openai_api_version
        openai.api_base = cfg.openai_api_base

        print("=========================================")
        for msg in messages:
            print(Fore.BLUE + f"{msg['role']}: {msg['content']}")
            print("-----------------------------------------")
        print("=========================================")


        response = openai.ChatCompletion.create(
            # deployment_id=cfg.openai_deployment_id,
            engine=cfg.openai_deployment_id,
            # model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
    else:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

    print(Fore.GREEN + response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content']
    # return response.choices[0].message["content"]
