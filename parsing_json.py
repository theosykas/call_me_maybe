from pydantic import BaseModel, Field
from typing import Dict, Optional
import json
import os


class ParamInput(BaseModel):
    type: str = Field(min_length=1)


class FunctionDefinition(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, ParamInput] = Field(min_length=1)
    returns: Dict[str, str] = Field(min_length=1)


class JsonParser:
    def __init__(self, path: Optional[str]) -> None:
        self.path = path

    def read_json(self) -> None:
        try:
            with open(self.path, "r") as file:
                data = json.load(file)
                return data  # write in ouput
        except FileNotFoundError:
            print("Error file is not found")
        except json.JSONDecodeError:
            print("invalid format of json file")

    # makedirs / directory nested(imbriquer)
    def create_output(self, data_saving: str) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)  # check avec exist_ok si le files exist
            with open(self.path, "w", encoding="UTF-8") as output_file:
                data_output = json.dump(data_saving, output_file, indent=4)
        except OSError:
            print(f"Directory {self.path} can not be created")

# __enter__ overhide with ([\n)
# __exit__ (]\n)
