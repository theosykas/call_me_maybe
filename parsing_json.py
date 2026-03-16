from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Dict, Optional, List
from pydantic import BaseModel, Field
import json
import os


class ParamInput(BaseModel):
    type: str = Field(min_length=1)


class FunctionDefinition(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, ParamInput] = Field(min_length=1)
    returns: Dict[str, str] = Field(min_length=1)

    @staticmethod
    def constrain_decoding(llm: Small_LLM_Model, user_prompt: str,
                           allowed_token: List[int]) -> List[int]:
        # encode le prompt[0]
        input_id = llm.encode(user_prompt).tolist()[0]  # tokenization of prompt tolist() == list cpy
        logits_list = llm.get_logits_from_input_ids(input_id)  # recuperer tout les lgits brut
        # logits_list = [1, 456, 54]
        vocab_size = len(logits_list)
        for mask in range(vocab_size):
            if mask not in allowed_token:
                logits_list[mask] = -float("inf")  # ecrase les scores par -inf
            return logits_list
        max_logits = max(logits_list)  # trouve le plus proche du max
        next_token_id = logits_list.index(max_logits)
        return next_token_id


class JsonParser:
    def __init__(self, path: Optional[str]) -> None:
        self.path = path

    def read_json(self) -> List[str]:
        try:
            with open(self.path, "r") as file:
                data = json.load(file)
                return data  # write in ouput
        except FileNotFoundError:
            print("Error file is not found")
        except json.JSONDecodeError:
            print("invalid format of json file")
        return None

    # makedirs / directory nested(imbriquer)
    # (dumps) -- create format json
    def create_json(self, data_write: str) -> List[str]:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="UTF-8") as output_file:
                data_output = json.dump(data_write, output_file, indent='\t')
                return data_output
        except OSError:
            print(f"Directory {self.path} can not be created")
        return None

# __enter__ overhide with ([\n)
# __exit__ (]\n)

# -inf == (-float('inf')) == 0% float - inf
