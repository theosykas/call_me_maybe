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
    function_reader = JsonParser(args.functions_definition)  # default value
    read_functions = function_reader.read_json()
    if args.input:
        test_reader = JsonParser(args.input)
        read_test = test_reader.read_json()


if __name__ == "__main__":
    main()
