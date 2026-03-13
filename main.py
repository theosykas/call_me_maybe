from parsing_json import JsonParser
import argparse


def main():
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
    output_created = output_create.create_output(read_functions)


if __name__ == "__main__":
    main()
