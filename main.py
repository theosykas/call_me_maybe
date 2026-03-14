from parsing_json import JsonParser
from init_sdk import init_llm
import argparse


def main():
    qwen_model = init_llm(model_name="Qwen/Qwen3-0.6B")
    print('charged qwen model llm_sdk\n')
    tokens = qwen_model.encode("hello World\n")
    print(f'token test {tokens}')
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
    function_json = JsonParser(args.functions_definition)  # default value
    read_functions = function_json.read_json()
    reader_json = JsonParser(args.input)
    read_test = reader_json.read_json()
    output_create = JsonParser(args.output)
    output_created = output_create.create_json(read_test)


# writer = JsonParser(args.output)
# writer.create_json(final_output)

if __name__ == "__main__":
    main()
