# Author: 赩林, xilin0x7f@163.com
import argparse

def arg_extractor(func):
    def wrapper(args):
        func_args = {
            k: v for k, v in vars(args).items()
            if k in func.__code__.co_varnames
        }
        return func(**func_args)
    return wrapper

@arg_extractor
def command1(input, output):
    """Handle the first subcommand."""
    print(f"Running command1 with input: {input} and output: {output}")

@arg_extractor
def command2(input, param):
    """Handle the second subcommand."""
    print(f"Running command2 with input: {input} and parameters: {param}")

@arg_extractor
def command3(input, output, i=10):
    print(f"Running command3 with input: {input} and output: {output}, i: {i}")

def main():
    parser = argparse.ArgumentParser(description="A script supporting multiple subcommands.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to run")

    parser_command1 = subparsers.add_parser("command1", help="Run the first command")
    parser_command1.add_argument("-i", "--input", required=True, help="Input file for command1")
    parser_command1.add_argument("-o", "--output", required=True, help="Output file for command1")
    parser_command1.set_defaults(func=command1)

    parser_command2 = subparsers.add_parser("command2", help="Run the second command")
    parser_command2.add_argument("-i", "--input", required=True, help="Input file for command2")
    parser_command2.add_argument("-p", "--param", type=int, required=True, help="Parameter for command2")
    parser_command2.set_defaults(func=command2)

    parser_command3 = subparsers.add_parser("command3", help="Run the second command")
    parser_command3.set_defaults(func=command3)
    parser_command3.add_argument("input", help="Input file for command2")
    parser_command3.add_argument("output", type=int, help="Parameter for command2")
    parser_command3.add_argument("-i", type=int, help="Parameter for command2")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
