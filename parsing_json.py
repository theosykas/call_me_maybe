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
    def get_allowed_token(prefix: str,
                          valid_token: List[int],
                          token_map: Dict[int, str],
                          mode: str = "string") -> List[int]:
        allowed_token = []
        for token_id, token_str in token_map.items():  # lit pre calcul dict (map)
            if mode == "number":
                numeric_value = set("0123456789.")
                for c in token_str:
                    if c not in numeric_value:
                        break
                else:
                    allowed_token.append(token_id)
            elif mode == "string":
                allowed_token.append(token_id)
            else:
                corresponding = False  # mis a false a chaques iterations
                combined = prefix + token_str
                for choice in valid_token:
                    if choice.startswith(combined):
                        corresponding = True
                        break
                if corresponding:
                    allowed_token.append(token_id)  # add token id to set allowed token
                    continue  # -> next_token
                if token_str == '"':  # end of word
                    for choice in valid_token:
                        if prefix == choice:
                            allowed_token.append(token_id)
                            break
        return allowed_token

    @staticmethod
    def constrain_decoding(
        llm: Small_LLM_Model,
        current_prefix: str,
        input_ids: List[int],
        allowed_token: List[str],
        token_map: Dict[int, str],
            mode: str = "string") -> List[int]:
        logits_list = llm.get_logits_from_input_ids(
            input_ids
        )  # recuperer tout les logits brut
        # logits_list = [1, 456, 54]
        allowed_token = set(allowed_token)
        token_selection = FunctionDefinition.get_allowed_token(
            current_prefix, allowed_token, token_map, mode)
        vocab_size = len(logits_list)
        for mask in range(vocab_size):
            if mask not in token_selection:
                logits_list[mask] = -float("inf")  # ecrase les scores par -inf
        max_logits = max(logits_list)  # trouve le plus proche du max
        next_token_id = logits_list.index(max_logits)
        return next_token_id

    @staticmethod
    def format_argument(args: Dict[str, ParamInput]) -> str:
        return str(list(args.keys()))

    # text_to_prompt = [0] fn.name = fn_add_nb fn.description = fn.decription args = [a, b]
    @staticmethod
    def function_catalog(functions: List["FunctionDefinition"]) -> str:
        catalog_function = ""
        for i, fn in enumerate(functions):
            args_list = FunctionDefinition.format_argument(fn.parameters)
            catalog_function += f"ID: [{i}] | FUNCTION_NAME '{fn.name}'\n"
            catalog_function += f"DESCRIPTION: {fn.description}\n"
            catalog_function += f"Args: {args_list}\n"
        return catalog_function

    # @staticmethod
    # def check_brace(count_brace: int, token_str: str) -> int:
    #     count_brace += token_str.count("{")
    #     count_brace -= token_str.count("}")
    #     return count_brace

    @staticmethod
    def generate_constrain_json(
        llm: Small_LLM_Model,
        user_prompt: str,
        catalog: str,
        functions_reader: List["FunctionDefinition"],
            token_map: Dict[int, str]) -> str:
        # user_escape = user_prompt.replace('"', '\\"')
        formated_ouput = f'{{"prompt": "{user_prompt}", "name": "'
        systeme_prompt = (
            "### Role\n"
            "You are an intelligent API Router. You map natural language "
            "requests to the correct Function ID.\n\n"
            "### Available Tools\n"
            f"{catalog}\n"
            "### User Request\n"
            f'"{user_prompt}"\n\n'
            "### Task\n"
            "1. Analyze the verbs and nouns in the User Request.\n"
            "2. Find the 'Action' in the Available Tools "
            "list that matches the request.\n"
            "3. Select the corresponding ID number in catalog.\n\n"
            "### Output\n"
            f"{formated_ouput}"
        )
        current_input = llm.encode(systeme_prompt + formated_ouput).tolist()[0]
        generate_ids = []
        current_prefix = ""
        fn_names = [fn.name for fn in functions_reader]
        max_token = 64
        for _ in range(max_token):
            next_id = FunctionDefinition.constrain_decoding(
                llm, current_prefix, current_input, fn_names, token_map, mode="catalog"
            )
            current_input.append(next_id)
            generate_ids.append(next_id)
            last_token = llm.decode([next_id])
            if last_token == '"':
                break
            current_prefix += last_token
        chose_function = next((f for f in functions_reader if f.name == current_prefix), None)
        if chose_function:
            param_bridge = ', "parameters": {'
            bridge_ids = llm.encode(param_bridge).tolist()[0]
            current_input.extend(bridge_ids)
            generate_ids.extend(bridge_ids)
            params = list(chose_function.parameters.items())
            for i, (p_name, p_info) in enumerate(params):
                key_str = f' "{p_name}": '
                if p_info.type == "string":
                    key_str += '"'
                key_ids = llm.encode(key_str).tolist()[0]
                current_input.extend(key_ids)
                generate_ids.extend(key_ids)
                current_prefix = ""
                val_mode = p_info.type
                # number_max_token = 16
                for _ in range(max_token):
                    next_id = FunctionDefinition.constrain_decoding(
                        llm, current_prefix, current_input, [], token_map, mode=val_mode
                    )
                    current_input.append(next_id)
                    generate_ids.append(next_id)
                    token_str = llm.decode([next_id])
                    current_prefix += token_str
                    if val_mode == "string" and token_str == '"':
                        break
                    if val_mode == "number" and "." in current_prefix and current_prefix[-1].isdigit():
                        break
                if i < len(params) - 1:
                    sep = ", "
                    sep_ids = llm.encode(sep).tolist()[0]
                    current_input.extend(sep_ids)
                    generate_ids.extend(sep_ids)
        closing = "}}"
        close_ids = llm.encode(closing).tolist()[0]
        current_input.extend(close_ids)
        generate_ids.extend(close_ids)
        return formated_ouput + llm.decode(generate_ids)


class JsonWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        self.json_file = None
        self.coma = True

    def __enter__(self) -> "JsonWriter":
        self.json_file = open(self.path, "w")
        self.json_file.write("[\n")
        return self

    def __exit__(self, exec_t, exec_v, exec_tb) -> None:
        if self.json_file:
            self.json_file.write("\n]")
            self.json_file.close()
        return self

    def write_json(self, data: Dict) -> None:
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
