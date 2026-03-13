from pydantic import BaseModel, Field
from typing import Dict, Optional
import json


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
                print("file data", data)
        except FileNotFoundError:
            print("Error file is not found")
        except json.JSONDecodeError:
            print("invalid format of json file")

    # def create_output(self) -> None:


# __enter__ overhide with ([\n)
# __exit__ (]\n)
