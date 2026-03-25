from llm_sdk.llm_sdk import Small_LLM_Model
from typing import Dict, Optional, List
from pydantic import BaseModel, Field
import json
import os


class TypePrompt(BaseModel):
    prompt: str = Field(min_length=1)


class ParamInput(BaseModel):
    """
    A Pydantic model representing a parameter input.

    Attributes:
        type (str): The type of the parameter. Must be a non-empty string.
    """
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
        """
    Filter tokens based on allowed characters and patterns for
    constrained text generation.

    This static method filters a token map to return only token
    IDs that meet specific
    constraints depending on the parsing mode. It's used to restrict token
    generation to
    valid continuations based on the current prefix and valid token options.

    Args:
        prefix (str): The current text prefix being built up during generation.
        valid_token (List[int]): A list of valid token strings or choices that
        can follow.
        token_map (Dict[int, str]): A mapping of token IDs to their string
        representations.
        mode (str, optional): The filtering mode. Defaults to "string".
            - "number": Only allows tokens
            containing numeric characters (0-9.-).
            - "string": Filters tokens for JSON string parsing,
            handling quotes and escapes.
            - "regex": Filters tokens for regex patterns,
            excluding special characters.
            - Other: Filters tokens that are valid prefixes of
            choices in valid_token.

        Returns:
        List[int]: A list of token IDs that satisfy the constraints of the
        given mode.

        Examples:
        - In "number" mode, filters to only numeric tokens.
        - In "string" mode, handles JSON string escaping and quote validation.
        - In "regex" mode, excludes regex special characters and newlines.
        - In default mode, ensures tokens are valid continuations of
        valid_token choices.
        """
        allowed_token = []
        stop_token = {',', ' ', '}'}
        for token_id, token_str in token_map.items():  # lit pre calcul dict (map)
            if mode == "number":
                numeric_value = set("0123456789.-")
                valid_nb = True
                for c in token_str:
                    if c not in numeric_value:
                        valid_nb = False
                        break
                if valid_nb:
                    allowed_token.append(token_id)
                elif token_str in stop_token:
                    allowed_token.append(token_id)
            elif mode == "integer":
                int_value = set("0123456789-")
                valid_int = True
                for c in token_str:
                    if c not in int_value:
                        valid_int = False
                        break
                if valid_int:
                    allowed_token.append(token_id)
                elif token_str in stop_token:
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
                if '"' not in token_str:  # autorise tout les token
                    allowed_token.append(token_id)
                elif token_str == '"':  # dernier guillemet seul "
                    allowed_token.append(token_id)
                elif token_str == '\\"':
                    if prefix.endswith("\\") and not prefix.endswith("\\\\"):
                        allowed_token.append(token_id)
            elif mode == "regex":
                if "\n" in token_str or "\r" in token_str:
                    continue
                if (
                    " " in token_str
                    or "|" in token_str
                    or "." in token_str
                    or "*" in token_str
                ):
                    continue
                forbiden = set(["\\", "\n", '"'])
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
                    allowed_token.append(token_id)  # add token id
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
        token_selection = set(
            FunctionDefinition.get_allowed_token(
                current_prefix,
                allowed_token,
                token_map,
                mode
            )
        )
        vocab_size = len(logits_list)
        for mask in range(vocab_size):
            if mask not in token_selection:
                logits_list[mask] = -float("inf")
        quotes_ids = None
        for token_id, token_str in token_map.items():
            if token_str == '"':
                quotes_ids = token_id
                break
        if quotes_ids is not None:
            if mode == "regex" and current_prefix.endswith(")"):
                logits_list[quotes_ids] += 50.0
        max_logits = max(logits_list)
        next_token_id = logits_list.index(max_logits)
        return next_token_id

    @staticmethod
    def format_argument(args: Dict[str, ParamInput]) -> str:
        """
        Format the argument dictionary keys into a string representation.

        Args:
            args: A dictionary mapping argument names to their
            ParamInput values.

        Returns:
            A string representation of the dictionary keys as a list.
        """
        return str(list(args.keys()))

    # text_to_prompt = [0] fn.name = fn_add_nb fn.description = fn.decription args = [a, b]
    @staticmethod
    def function_catalog(functions: List["FunctionDefinition"]) -> str:
        """
        Generate a formatted catalog of function definitions.

        Creates a human-readable string representation
        of multiple function definitions,
        including their index, name, description, and arguments.

        Args:
            functions: A list of FunctionDefinition objects to be cataloged.

        Returns:
            A formatted string containing the catalog
            of all functions, where each function
            is represented with its ID, name,
            description, and arguments on separate lines.

        Example:
            If given two functions, the output would be:
            ID: [0] | FUNCTION_NAME 'function_name'
            DESCRIPTION: Function description
            Args: arg1: type, arg2: type
            ID: [1] | FUNCTION_NAME 'another_function'
            DESCRIPTION: Another description
            Args: arg3: type
        """
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
        user_escape = user_prompt.replace("\\", "\\\\").replace('"', '\\"')
        formated_ouput = f'{{"prompt": "{user_escape}", "name": "'
        systeme_prompt = (
            "### ROLE\n"
            "You are a High-Precision Value Extractor. Your task is to"
            "identify the correct function and provide the exact raw values for its parameters based on the User Request.\n\n"
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
            bridge_ids = llm.encode(param_bridge).tolist()[0]
            current_input.extend(bridge_ids)
            generate_ids.extend(bridge_ids)
            params = list(chose_fn.parameters.items())
            for i, (p_name, p_info) in enumerate(params):  # boucle sur les enum param name: string
                # prefix = "{" if i == 0 else ', '
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
                replace_dict = {
                    "asterisk": "*",
                    "dote": ".",
                    "underscore": "_",
                    "dash": "-",
                    "space": " "
                }
                for _ in range(max_token):  # str or number
                    next_id = FunctionDefinition.constrain_decoding(
                        llm,
                        current_prefix,
                        current_input,
                        valid_source,
                        token_map,
                        mode=val_mode,
                    )
                    current_input.append(next_id)
                    # generate_ids.append(next_id)
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
                    if val_mode == "number":
                        # after_dote = current_prefix.split(".")
                        # if len(after_dote) > 1 and len(after_dote[-1]) >= 1:
                        if token_str in {',', '}', ' '}:
                            if '.' not in current_prefix:
                                current_prefix += ".0"
                            terminated = True
                            break
                        current_prefix += token_str
                    if val_mode == "integer":
                        if token_str in {',', '}', ' '}:
                            terminated = True
                            break
                        current_prefix += token_str
                output = current_prefix
                clean = output.strip('"').lower()
                if p_name == "replacement" and clean in replace_dict:
                    output = output.lower().replace(clean, replace_dict[clean])
                if (
                    not terminated
                    and val_mode in ("string", "regex")
                    and token_str == '"'
                ):
                    quotes = '"'
                    quotes_ids = llm.encode(quotes).tolist()[0]
                    current_input.extend(quotes_ids)
                    generate_ids.extend(quotes_ids)
                    # output += '"'
                val_ids = llm.encode(output).tolist()[0]
                generate_ids.extend(val_ids)
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
        """
        Initialize the JSON parser object.

        Args:
            path (str): The file path to the JSON file to be parsed.

        Attributes:
            path (str): The file path to the JSON file.
            json_file (None): Placeholder for the loaded JSON file content.
            coma (bool): Flag indicating whether
            commas are present (default: True).
        """
        self.path = path
        self.json_file = None
        self.coma = True

    def __enter__(self) -> "JsonWriter":
        """
        Enter the context manager and initialize the JSON writer.

        Opens the file at the specified path in write mode and
        writes the opening
        bracket of a JSON array. Returns the context manager
        instance for use in the `with` statement.

        Returns:
            JsonWriter: The context manager instance (self).

        Raises:
            IOError: If the file cannot be opened at the specified path.
        """
        self.json_file = open(self.path, "w")
        self.json_file.write("[\n")
        return self

    def __exit__(self, exec_t, exec_v, exec_tb) -> None:
        """
        Exit the context manager and finalize the JSON file.

        Closes the JSON file by writing a closing bracket
        and closing the file handle.
        This method is called when exiting a `with` statement.

        Args:
            exec_t: The exception type, if any exception
            occurred in the with block.
            exec_v: The exception instance, if any exception
            occurred in the with block.
            exec_tb: The exception traceback, if any exception
            occurred in the with block.

        Returns:
            None
        """
        if self.json_file:
            self.json_file.write("\n]")
            self.json_file.close()
        return self

    def write_json(self, data: Dict) -> None:
        """
        Write JSON data to the file.

        If this is not the first write operation, prepends a comma and newline
        to separate from previous entries. Writes the data as formatted JSON
        with tab indentation.

        Args:
            data (Dict): Dictionary containing the data to be serialized
            and written as JSON.
        """
        if not self.coma:
            self.json_file.write(",\n")
        json_data = json.dumps(data, indent="\t")
        self.coma = False
        self.json_file.write(json_data)


class JsonParser:
    def __init__(self, path: Optional[str]) -> None:
        """
        Initialize the instance with an optional file path.
        Args:
            path (Optional[str]): The file path to be used.
            Can be None if no path is provided.
        Returns:
            None
        """
        self.path = path

    def read_json(self) -> List[str]:
        """
        Read and parse a JSON file from the specified file path.

        Attempts to open and load a JSON file from the
        path stored in self.path.
        Returns the parsed JSON data if successful.

        Returns:
            List[str]: The parsed JSON data from the file,
            or None if an error occurs.

        Raises (handled internally):
            FileNotFoundError: If the file at self.path does not exist.
                Prints an error message and returns None.
            json.JSONDecodeError: If the file contains invalid JSON format.
                Prints an error message and returns None.
        """
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
        """
        Create and write JSON data to a file at the specified path.

        This method creates any necessary parent directories
        and writes the provided
        data to a JSON file with UTF-8 encoding and tab indentation.

        Args:
            data_write (str): The data to be written to the JSON file.

        Returns:
            List[str]: The result of json.dump()
            operation (typically None on success),
            or None if an OSError occurs during directory
            creation or file writing.
        Raises:
            Prints an error message if the directory
            cannot be created due to OSError.
        Note:
            - The method creates parent directories
            recursively if they don't exist.
            - Files are written with UTF-8 encoding and tab indentation.
            - Consider the return type annotation (List[str]) as it may
            not accurately
              reflect the actual return value from json.dump().
        """
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="UTF-8") as output_file:
                data_output = json.dump(data_write, output_file, indent="\t")
                return data_output
        except OSError:
            print(f"Directory {self.path} can not be created")
        return None

# /   {
# //     "prompt": "Calculate the square root of 144"
# //   }
