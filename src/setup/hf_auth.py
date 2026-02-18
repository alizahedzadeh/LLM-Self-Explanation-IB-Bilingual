import os
from dotenv import load_dotenv
from huggingface_hub import login

def hf_login():
    load_dotenv()  # load .env file
    token = os.getenv("HF_TOKEN")

    if token is None:
        raise ValueError("HF_TOKEN not found in environment variables")

    login(token)
