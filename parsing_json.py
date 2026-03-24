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
    def get_allowed_token(
        prefix: str,
        valid_token: List[int],
        token_map: Dict[int, str],
        mode: str = "string",
    ) -> List[int]:
        allowed_token = []
        for token_id, token_str in token_map.items():  # lit pre calcul dict (map)
            if mode == "number":
                numeric_value = set("0123456789.-")
                for c in token_str:
                    if c not in numeric_value:
                        break
                else:
                    allowed_token.append(token_id)
            elif mode == "string":
                if valid_token:
                    token_list = list(valid_token)
                    prompt_token = True
                    allowed_char = set(token_list[0] + ' \\"Ġ ')
                    for c in token_str:
                        if c not in allowed_char:
                            prompt_token = False
                            break
                    if not prompt_token:
                        continue  # Rejette le token s'il contient des lettres hors prompt
                if prefix == "":
                    if token_str.startswith(" "):
                        continue
                # if prefix.count('*') + token_str.count('*') > 1:
                #     continue
                if '"' not in token_str:  # autorise tout les token
                    allowed_token.append(token_id)
                elif token_str == '"':  # dernier guillemet seul "
                    allowed_token.append(token_id)
                elif token_str == '\\"':
                    if prefix.endswith('\\') and not prefix.endswith('\\\\'):
                        allowed_token.append(token_id)
            elif mode == "regex":
                if "\n" in token_str or "\r" in token_str:
                    continue
                if " " in token_str or '|' in token_str or '.' in token_str or '*' in token_str:
                    continue
                forbiden = set(['\\', '\n', '"'])
                valid = True
                for c in forbiden:
                    if c in token_str:
                        valid = False
                        break
                if valid:
                    allowed_token.append(token_id)
                if token_str == '"':  # autorise token close "
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
                if token_str == '"':  # end of word nom complet
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
        mode: str = "string",
    ) -> List[int]:
        logits_list = llm.get_logits_from_input_ids(
            input_ids
        )  # recuperer tout les logits brut
        # logits_list = [1, 456, 54]
        allowed_token = set(allowed_token)
        token_selection = set(FunctionDefinition.get_allowed_token(
            current_prefix, allowed_token, token_map, mode
        ))
        vocab_size = len(logits_list)
        for mask in range(vocab_size):
            if mask not in token_selection:
                logits_list[mask] = -float("inf")  # ecrase les scores par -inf
        quotes_ids = None
        for token_id, token_str in token_map.items():
            if token_str == '"':
                quotes_ids = token_id
                break
        if quotes_ids is not None:
            if mode == "regex" and current_prefix.endswith(')'):
                logits_list[quotes_ids] += 50.0  # augmente la probabilité de fermeture 
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

    @staticmethod
    def generate_constrain_json(
        llm: Small_LLM_Model,
        user_prompt: str,
        catalog: str,
        functions_reader: List["FunctionDefinition"],
        token_map: Dict[int, str],
    ) -> str:
        user_escape = user_prompt.replace('\\', '\\\\').replace('"', '\\"')
        formated_ouput = f'{{"prompt": "{user_escape}", "name": "'
        # systeme_prompt = (
        #     "### Role\n"
        #     "You are an API Router. You extract parameters from User Requests to build a valid JSON.\n\n"
        #     "### Available Tools\n"
        #     f"{catalog}\n"
        #     "### Regex Rules\n"
        #     "You MUST write the shortest and most accurate regex possible. NEVER use '.*' or wildcards. Use exact syntax:\n"
        #     "- vowels → ([aeiouAEIOU])\n"
        #     "- digits → ([0-9]+)\n"
        #     "- exact word → the word itself (e.g. cat)\n\n"
        #     "### User Request\n"
        #     f'"{user_escape}"\n\n'
        #     "### Task\n"
        #     "1. Find the correct Action.\n"
        #     "2. Extract the exact parameters.\n\n"
        #     "### Output\n"
        #     f"{formated_ouput}"
        # )
        systeme_prompt = (
            "### ROLE\n"
            "You are a High-Precision Value Extractor. Your task is to identify the correct function and provide the exact raw values for its parameters based on the User Request.\n\n"
            "### TOOL CATALOG\n"
            f"{catalog}\n\n"
            "### REGEX GENERATION RULES\n"
            "When a regex is required, you must provide ONLY the pattern using this strict syntax:\n"
            "- To match numbers/digits: ([0-9]+)\n"
            "- To match vowels: ([aeiouAEIOU])\n"
            "- To match a specific word: use the word itself (e.g., 'cat')\n"
            "CRITICAL: Never use '.*' or broad wildcards. Be as specific as possible.\n\n"
            "### EXTRACTION TASK\n"
            "1. Identify the Function Name from the catalog.\n"
            "2. Extract or generate the specific string, number, or regex pattern for each parameter.\n\n"
            "### USER REQUEST\n"
            f'"{user_prompt}"\n\n'
            "### VALUE EXTRACTION\n"
            "### Output\n"
            f"{formated_ouput}"
        )
        current_input = llm.encode(systeme_prompt).tolist()[0]
        generate_ids = []
        current_prefix = ""
        fn_names = [fn.name for fn in functions_reader]  # fn_greet ...
        max_token = 64
        chose_fn = None
        for _ in range(max_token):
            next_id = FunctionDefinition.constrain_decoding(  # seul token == fn_name
                llm,
                current_prefix,
                current_input,
                fn_names,
                token_map,
                mode="catalog"
            )
            current_input.append(next_id)
            generate_ids.append(next_id)  # token fn_name
            last_token = llm.decode([next_id])
            if last_token == '"':  # last '"'
                break
            current_prefix += last_token  # == fn_greet
        for fn in functions_reader:
            if fn.name != current_prefix:
                continue
            chose_fn = fn
            break
        if chose_fn:
            param_bridge = ', "parameters": {'
            # param_bridge[-1]
            bridge_ids = llm.encode(param_bridge).tolist()[0]
            current_input.extend(bridge_ids)
            generate_ids.extend(bridge_ids)
            params = list(chose_fn.parameters.items())
            for i, (p_name, p_info) in enumerate(params):  # boucle sur les enum param name: string
                key_str = f' "{p_name}": '
                val_mode = p_info.type
                if p_name == "regex":
                    val_mode = "regex"
                if val_mode in ("string", "regex"):
                    key_str += '"'  # ouvre les "
                key_ids = llm.encode(key_str).tolist()[0]
                current_input.extend(key_ids)
                generate_ids.extend(key_ids)
                current_prefix = ""
                terminated = False
                if val_mode == "string":
                    valid_source = [user_prompt]
                else:
                    valid_source = []
                # valid_source = [user_prompt] if val_mode == "string" else []
                for _ in range(max_token):  # str or number
                    next_id = FunctionDefinition.constrain_decoding(
                        llm,
                        current_prefix,
                        current_input,
                        valid_source,
                        token_map,
                        mode=val_mode
                    )
                    current_input.append(next_id)
                    generate_ids.append(next_id)
                    token_str = llm.decode([next_id])
                    if val_mode in ("string", "regex"):  # and token_str == last_token:
                        backslash_count = 0
                        for char in reversed(current_prefix):
                            if char == "\\":
                                backslash_count += 1
                            else:
                                break
                        current_prefix += token_str  # accumule
                        if current_prefix.endswith('"'):
                            if backslash_count % 2 == 0:
                                terminated = True
                                break
                    if val_mode == "number" and token_str in (',', '}', ' '):
                        current_prefix += token_str
                        after_dote = current_prefix.split(".")
                        if len(after_dote) > 1 and len(after_dote[-1]) >= 1:
                            terminated = True
                            break
                    # on stop sur un nb complet ou sur un "
                if not terminated and val_mode in ("string", "regex") and token_str == '"':
                    quotes = '"'
                    quotes_ids = llm.encode(quotes).tolist()[0]
                    current_input.extend(quotes_ids)
                    generate_ids.extend(quotes_ids)
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


# constrain_char = ['+', ']', ')']
                # pattern_finished = False
                # for pattern in constrain_char:
                #     if prefix.endswith(pattern):
                #         pattern_finished = True
                #         break
                # open_parenthesis = prefix.count('(')
                # close_parenthesis = prefix.count(')')
                # if pattern_finished:
                #     if open_parenthesis > close_parenthesis:
                #         if token_str == ')':
                #             allowed_token.append(token_id)
                #     else:
                #         if token_str == '"':
                #             allowed_token.append(token_id)
                # else:

                # Return ONLY one regex. Do not repeat patterns.