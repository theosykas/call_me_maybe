from parsing_json import JsonParser, JsonWriter, FunctionDefinition
from init_sdk import init_llm
import argparse
import json


def main() -> None:
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
    catalog = FunctionDefinition.function_catalog(functions_reader)  # tr en catalog pour qwen
    generate_output = JsonParser(args.input).read_json()
    voc_size = len(qwen_model.get_logits_from_input_ids(
        qwen_model.encode(" ").tolist()[0]))  # generation all ids token
    token_map = {i: qwen_model.decode([i]) for i in range(voc_size)}
    # valid_choice = [fn.name for fn in functions_reader]
    JsonParser(args.output).create_ouptut([])
    with JsonWriter(args.output) as write:
        for case_input in generate_output:
            user_prompt = case_input["prompt"]
            generate_json = FunctionDefinition.generate_constrain_json(
                llm=qwen_model,
                user_prompt=user_prompt,
                catalog=catalog,
                functions_reader=functions_reader,
                token_map=token_map
            )
            try:
                data_dict = json.loads(generate_json)
                print('valid Json\n\n')
                print(generate_json)
                write.write_json(data_dict)
            except Exception as e:
                print(f'jsondecode Error {e}\n')
                print(generate_json)


if __name__ == "__main__":
    main()
