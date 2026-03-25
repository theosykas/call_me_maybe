from parsing_json import JsonParser, JsonWriter, FunctionDefinition, TypePrompt
from pydantic import ValidationError
from init_sdk import init_llm
import argparse
import json


def main() -> None:
    try:
        qwen_model = init_llm(model_name="Qwen/Qwen3-0.6B")
        parse_arg = argparse.ArgumentParser()
        parse_arg.add_argument(
            "--functions_definition",
            default="data/input/functions_definition.json"
        )
        parse_arg.add_argument(
            "--input",
            default="data/input/function_calling_tests.json"
        )
        parse_arg.add_argument(
            "--output",
            default="data/output/function_calls.json"
        )
        args = parse_arg.parse_args()
        raw_defs = JsonParser(args.functions_definition).read_json()
        functions_reader = [FunctionDefinition(**d) for d in raw_defs]
        catalog = FunctionDefinition.function_catalog(functions_reader)
        generate_output = JsonParser(args.input).read_json()
        voc_size = len(qwen_model.get_logits_from_input_ids(
            qwen_model.encode(" ").tolist()[0]))
        token_map = {i: qwen_model.decode([i]) for i in range(voc_size)}
        JsonParser(args.output).create_ouptut([])
        with JsonWriter(args.output) as write:
            for case_input in generate_output:
                try:
                    validate_case = TypePrompt(**case_input)
                    user_prompt = validate_case.prompt
                    generate_json = FunctionDefinition.generate_constrain_json(
                        llm=qwen_model,
                        user_prompt=user_prompt,
                        catalog=catalog,
                        functions_reader=functions_reader,
                        token_map=token_map
                    )
                    data_dict = json.loads(generate_json)
                    write.write_json(data_dict)
                    print(generate_json)
                except ValidationError as e:
                    for e in e.errors():
                        print(e["msg"])
                except json.JSONDecodeError:
                    print('invalid json generated\n')
                    print(generate_json)
                except Exception:
                    print('Error unexpected\n')
    except Exception as e:
        for e in e.errors():
            print(e)


if __name__ == "__main__":
    main()
