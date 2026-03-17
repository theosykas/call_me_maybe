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
    def constrain_decoding(
        llm: Small_LLM_Model, input_ids: List[int], allowed_token: List[int]
    ) -> List[int]:
        logits_list = llm.get_logits_from_input_ids(
            input_ids
        )  # recuperer tout les logits brut
        # logits_list = [1, 456, 54]
        allowed = set(allowed_token)  # la liste des tokens autorisés à être générés à ce moment précis 1 logits par token
        vocab_size = len(logits_list)
        for mask in range(vocab_size):
            if mask not in allowed:
                logits_list[mask] = -float("inf")  # ecrase les scores par -inf
        max_logits = max(logits_list)  # trouve le plus proche du max
        next_token_id = logits_list.index(max_logits)
        return next_token_id

    def format_argument(args: Dict[str, ParamInput]) -> str:
        return str(list(args.keys()))

    # text_to_prompt = [0] fn.name = fn_add_nb fn.description = fn.decription args = [a, b]
    def function_catalog(functions: List["FunctionDefinition"]) -> str:
        catalog_function = ""
        for i, fn in enumerate(functions):
            args_list = FunctionDefinition.format_argument(fn.parameters)
            catalog_function += f"[{i}] {fn.name}: {fn.description}\n"
            catalog_function += f"Args: {args_list}\n\n"
        return catalog_function

    def check_brace(count_brace: int, token_str: str) -> int:
        count_brace += token_str.count("{")
        count_brace -= token_str.count("}")
        return count_brace

    # LLM Response Prefixing == systeme_prompt
    def generate_constrain_json(
        llm: Small_LLM_Model, user_prompt: str, catalog: str, allowed_token: List[int]
    ) -> List[str]:
        formated_ouput = f'{{"prompt": "{user_prompt}", "name": "'
        # str start: '{' end: '}'
        systeme_prompt = (
            "### Role\n"
            "You are an intelligent API Router. Your goal is to map user requests "
            "to the exact function and arguments needed.\n\n"
            "### Available Catalog\n"
            f"{catalog}"
            "### Task\n"
            "Output a single JSON object with the keys 'prompt', 'name', and 'parameters'.\n\n"
            "### User Request\n"
            f'"{user_prompt}"\n\n'
            "### Output\n"
            f"{formated_ouput}"
        )
        current_input = llm.encode(systeme_prompt).tolist()[0]  # [1345, 49, 4907890...]
        generate_ids: List[str] = []
        max_token: int = 128
        brace_count = 1  # pour la premiere ouverte a 1
        for _ in range(max_token):
            next_id = FunctionDefinition.constrain_decoding(
                llm, current_input, allowed_token  # who is best -> token
            )
            current_input.append(next_id)
            generate_ids.append(next_id)
            last_token = llm.decode([next_id])
            brace_count = FunctionDefinition.check_brace(brace_count, last_token)
            if brace_count == 0:
                break
        generated_qwery = llm.decode(
            generate_ids
        )  # decode uniquement ce que on a generer
        final_output = formated_ouput + generated_qwery
        return final_output


class JsonWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        self.json_file = None
        self.coma = True  # char premier

    def __enter__(self) -> "JsonWriter":
        self.json_file = open(self.path, "w")
        self.json_file.write("[\n")
        return self

    def __exit__(self, exec_t, exec_v, exec_tb) -> "JsonWriter":
        if self.json_file:
            self.json_file.write("\n]")
            self.json_file.close()
        return self

    def write_json(self, data: Dict) -> "JsonWriter":
        if not self.coma:
            self.json_file.write(",\n")
        json_data = json.dumps(data, indent="\t")
        self.coma = False
        self.json_file.write(json_data)


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
    def create_ouptut(self, data_write: str) -> List[str]:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="UTF-8") as output_file:
                data_output = json.dump(data_write, output_file, indent="\t")
                return data_output
        except OSError:
            print(f"Directory {self.path} can not be created")
        return None


# -inf == (-float('inf')) == 0% float - inf
