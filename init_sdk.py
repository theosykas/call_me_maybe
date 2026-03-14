from llm_sdk.llm_sdk import Small_LLM_Model
# import torch


def init_llm(model_name: str):
    llm_qwen = Small_LLM_Model()
    return llm_qwen
