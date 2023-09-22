import getpass
import os
import platform
from pprint import pprint

import openai
import tokentrim as tt
from rich import print

from .cli import cli
from .code_block import CodeBlock
from .code_interpreter import CodeInterpreter
from .message_block import MessageBlock
from .utils import merge_deltas, parse_partial_json

function_schema = {
    "name": "run_code",
    "description": "Executes code on the user's machine and returns the output",
    "parameters": {
        "type": "object",
        "properties": {
            "language": {"type": "string", "description": "The programming language", "enum": ["python", "shell", "applescript", "javascript", "html"]},
            "code": {"type": "string", "description": "The code to execute"},
        },
        "required": ["language", "code"],
    },
}


class Interpreter:
    def __init__(self):
        self.messages = []
        self.temperature = 0.001
        self.local = False
        self.model = "gpt-4"
        self.debug_mode = False

        here = os.path.abspath(os.path.dirname(__file__))
        with open(os.path.join(here, "system_message.txt"), "r") as f:
            self.system_message = f.read().strip()

        # Store Code Interpreter instances for each language
        self.code_interpreters = {}

        self.active_block = None

    def cli(self):
        cli(self)

    def get_info_for_system_message(self):
        info = ""
        username = getpass.getuser()
        current_working_directory = os.getcwd()
        operating_system = platform.system()
        info += f"\n\n[User Info]\nName: {username}\nCWD: {current_working_directory}\nOS: {operating_system}"
        return info

    def chat(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]

        while True:
            try:
                user_input = input("> ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print()  # Aesthetic choice
                break

            self.messages.append({"role": "user", "content": user_input})

            try:
                self.respond()
            except KeyboardInterrupt:
                pass
            finally:
                # Always end the active block. Multiple Live displays = issues
                self.end_active_block()

    def end_active_block(self):
        if self.active_block:
            self.active_block.end()
            self.active_block = None

    def respond(self):
        info = self.get_info_for_system_message()
        system_message = self.system_message + "\n\n" + info

        messages = tt.trim(self.messages, self.model, system_message=system_message)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            functions=[function_schema],
            stream=True,
            temperature=self.temperature,
        )
        pprint("====== prompt begin ======")
        pprint(messages)
        pprint("====== prompt end ========")

        self.messages.append({})
        in_function_call = False
        self.active_block = None

        for chunk in response:
            delta = chunk["choices"][0]["delta"]
            self.messages[-1] = merge_deltas(self.messages[-1], delta)

            condition = "function_call" in self.messages[-1]
            if condition:
                if in_function_call is False:
                    self.end_active_block()

                    # Print newline if it was just a code block or user message
                    # (this just looks nice)
                    last_role = self.messages[-2]["role"]
                    if last_role == "user" or last_role == "function":
                        print()

                    self.active_block = CodeBlock()

                in_function_call = True

                if "arguments" in self.messages[-1]["function_call"]:
                    arguments = self.messages[-1]["function_call"]["arguments"]
                    new_parsed_arguments = parse_partial_json(arguments)
                    if new_parsed_arguments:
                        self.messages[-1]["function_call"]["parsed_arguments"] = new_parsed_arguments

            else:
                in_function_call = False
                if self.active_block is None:
                    self.active_block = MessageBlock()

            self.active_block.update_from_message(self.messages[-1])

            if chunk["choices"][0]["finish_reason"]:
                if chunk["choices"][0]["finish_reason"] != "function_call":
                    self.active_block.end()
                    return

                if chunk["choices"][0]["finish_reason"] == "function_call":
                    print("Running function:")
                    print(self.messages[-1])
                    print("---")

                    # End the active block so you can run input() below it
                    # Save language and code so we can create a new block in a moment
                    self.active_block.end()
                    language = self.active_block.language
                    code = self.active_block.code

                    response = input("  Would you like to run this code? (y/n)\n\n  ")
                    print("")  # <- Aesthetic choice

                    if response.strip().lower() == "y":
                        self.active_block = CodeBlock()
                        self.active_block.language = language
                        self.active_block.code = code
                    else:
                        self.active_block.end()
                        self.messages.append({"role": "function", "name": "run_code", "content": "User decided not to run this code."})
                        return

                    if not self.local and "parsed_arguments" not in self.messages[-1]["function_call"]:
                        self.messages.append(
                            {
                                "role": "function",
                                "name": "run_code",
                                "content": """Your function call could not be parsed. Please use ONLY the `run_code` function, which takes two parameters: `code` and `language`. Your response should be formatted as a JSON.""",
                            }
                        )
                        self.respond()
                        return

                    language = self.messages[-1]["function_call"]["parsed_arguments"]["language"]
                    if language not in self.code_interpreters:
                        self.code_interpreters[language] = CodeInterpreter(language, self.debug_mode)
                    code_interpreter = self.code_interpreters[language]

                    code_interpreter.active_block = self.active_block
                    code_interpreter.run()

                    self.active_block.end()

                    self.messages.append(
                        {"role": "function", "name": "run_code", "content": self.active_block.output if self.active_block.output else "No output"}
                    )

                    self.respond()
