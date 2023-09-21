from .cli import cli
from .utils import merge_deltas, parse_partial_json
from .message_block import MessageBlock
from .code_block import CodeBlock
from .code_interpreter import CodeInterpreter
from .llama_2 import get_llama_2_instance

import os
import time
import json
import platform
import openai
import getpass
import requests
import readline
import urllib.parse
import tokentrim as tt
from pprint import pprint
from rich import print
from rich.markdown import Markdown
from rich.rule import Rule

# Function schema for gpt-4
function_schema = {
  "name": "run_code",
  "description":
  "Executes code on the user's machine and returns the output",
  "parameters": {
    "type": "object",
    "properties": {
      "language": {
        "type": "string",
        "description":
        "The programming language",
        "enum": ["python", "shell", "applescript", "javascript", "html"]
      },
      "code": {
        "type": "string",
        "description": "The code to execute"
      }
    },
    "required": ["language", "code"]
  },
}

# Message for when users don't have an OpenAI API key.
missing_api_key_message = """> OpenAI API key not found

To use `GPT-4` (recommended) please provide an OpenAI API key.

To use `Code-Llama` (free but less capable) press `enter`.
"""

confirm_mode_message = """
**Open Interpreter** will require approval before running code. Use `interpreter -y` to bypass this.

Press `CTRL-C` to exit.
"""


class Interpreter:

  def __init__(self):
    self.messages = []
    self.temperature = 0.001
    self.api_key = None
    self.auto_run = False
    self.local = False
    self.model = "gpt-4"
    self.debug_mode = False

    # Get default system message
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'system_message.txt'), 'r') as f:
      self.system_message = f.read().strip()

    # Store Code Interpreter instances for each language
    self.code_interpreters = {}

    # No active block to start
    # (blocks are visual representation of messages on the terminal)
    self.active_block = None

    # Note: While Open Interpreter can use Llama, we will prioritize gpt-4.
    # gpt-4 is faster, smarter, can call functions, and is all-around easier to use.
    # This makes gpt-4 better aligned with Open Interpreters priority to be easy to use.
    self.llama_instance = None

  def cli(self):
    # The cli takes the current instance of Interpreter,
    # modifies it according to command line flags, then runs chat.
    cli(self)

  def get_info_for_system_message(self):
    info = ""
    username = getpass.getuser()
    current_working_directory = os.getcwd()
    operating_system = platform.system()
    info += f"\n\n[User Info]\nName: {username}\nCWD: {current_working_directory}\nOS: {operating_system}"

    # Open Procedures is an open-source database of tiny, structured coding tutorials.
    # We can query it semantically and append relevant tutorials/procedures to our system message:

    # Use the last two messages' content or function call to semantically search
    query = []
    for message in self.messages[-2:]:
      message_for_semantic_search = {"role": message["role"]}
      if "content" in message:
        message_for_semantic_search["content"] = message["content"]
      if "function_call" in message and "parsed_arguments" in message["function_call"]:
        message_for_semantic_search["function_call"] = message["function_call"]["parsed_arguments"]
      query.append(message_for_semantic_search)

    url = "https://open-procedures.replit.app/search/"

    pprint("===== query =====")
    pprint(query)

    try:
      relevant_procedures = requests.get(url, data=json.dumps(query)).json()["procedures"]
      pprint(relevant_procedures)
      info += "\n\n# Recommended Procedures\n" + "\n---\n".join(relevant_procedures) + "\nIn your plan, include steps and, if present, **EXACT CODE SNIPPETS** (especially for depracation notices, **WRITE THEM INTO YOUR PLAN -- underneath each numbered step** as they will VANISH once you execute your first line of code, so WRITE THEM DOWN NOW if you need them) from the above procedures if they are relevant to the task. Again, include **VERBATIM CODE SNIPPETS** from the procedures above if they are relevent to the task **directly in your plan.**"
    except Exception as e:
      pprint(requests.get(url, data=json.dumps(query)))
      pprint(e)
      # For someone, this failed for a super secure SSL reason.
      # Since it's not stricly necessary, let's worry about that another day. Should probably log this somehow though.
      pass

    return info

  def reset(self):
    self.messages = []
    self.code_interpreters = {}

  def load(self, messages):
    self.messages = messages

  def chat(self, message=None, return_messages=False):

    # Connect to an LLM (an large language model)
    # gpt-4
    self.verify_api_key()

    # Display welcome message
    welcome_message = ""

    if self.debug_mode:
      welcome_message += "> Entered debug mode"

    # If self.local, we actually don't use self.model
    # (self.auto_run is like advanced usage, we display no messages)
    if not self.local and not self.auto_run:
      welcome_message += f"\n> Model set to `{self.model.upper()}`\n\n**Tip:** To run locally, use `interpreter --local`"

    # If not auto_run, tell the user we'll ask permission to run code
    # We also tell them here how to exit Open Interpreter
    if not self.auto_run:
      welcome_message += "\n\n" + confirm_mode_message

    welcome_message = welcome_message.strip()

    # Print welcome message with newlines on either side (aesthetic choice)
    # unless we're starting with a blockquote (aesthetic choice)
    if welcome_message != "":
      if welcome_message.startswith(">"):
        print(Markdown(welcome_message), '')
      else:
        print('', Markdown(welcome_message), '')

    # Check if `message` was passed in by user
    if message:
      # If it was, we respond non-interactivley
      self.messages.append({"role": "user", "content": message})
      self.respond()

    else:
      # If it wasn't, we start an interactive chat
      while True:
        try:
          user_input = input("> ").strip()
        except EOFError:
          break
        except KeyboardInterrupt:
          print()  # Aesthetic choice
          break

        # Use `readline` to let users up-arrow to previous user messages,
        # which is a common behavior in terminals.
        readline.add_history(user_input)

        # Add the user message to self.messages
        self.messages.append({"role": "user", "content": user_input})

        # Let the user turn on debug mode mid-chat
        if user_input == "%debug":
            print('', Markdown("> Entered debug mode"), '')
            print(self.messages)
            self.debug_mode = True
            continue

        # Respond, but gracefully handle CTRL-C / KeyboardInterrupt
        try:
          self.respond()
        except KeyboardInterrupt:
          pass
        finally:
          # Always end the active block. Multiple Live displays = issues
          self.end_active_block()

    if return_messages:
        return self.messages

  def verify_api_key(self):
    """
    Makes sure we have an OPENAI_API_KEY.
    """

    if self.api_key == None:

      if 'OPENAI_API_KEY' in os.environ:
        self.api_key = os.environ['OPENAI_API_KEY']
      else:
        # This is probably their first time here!
        print('', Markdown("**Welcome to Open Interpreter.**"), '')
        time.sleep(1)

        print(Rule(style="white"))

        print(Markdown(missing_api_key_message), '', Rule(style="white"), '')
        response = input("OpenAI API key: ")

        if response == "":
            # User pressed `enter`, requesting Code-Llama
            self.local = True

            print(Markdown("> Switching to `Code-Llama`...\n\n**Tip:** Run `interpreter --local` to automatically use `Code-Llama`."), '')
            time.sleep(2)
            print(Rule(style="white"))
            return

        else:
            self.api_key = response
            print('', Markdown("**Tip:** To save this key for later, run `export OPENAI_API_KEY=your_api_key` on Mac/Linux or `setx OPENAI_API_KEY your_api_key` on Windows."), '')
            time.sleep(2)
            print(Rule(style="white"))

    openai.api_key = self.api_key

  def end_active_block(self):
    if self.active_block:
      self.active_block.end()
      self.active_block = None

  def respond(self):
    # Add relevant info to system_message
    # (e.g. current working directory, username, os, etc.)
    info = self.get_info_for_system_message()
    pprint("===== info =====")
    pprint(info)
    system_message = self.system_message + "\n\n" + info

    messages = tt.trim(self.messages, self.model, system_message=system_message)

    if self.debug_mode:
      print("\n", "Sending `messages` to LLM:", "\n")
      print(messages)
      print()

    # gpt-4
    response = openai.ChatCompletion.create(
      model=self.model,
      messages=messages,
      functions=[function_schema],
      stream=True,
      temperature=self.temperature,
    )
    pprint("====== prompt ======")
    pprint(messages)

    # Initialize message, function call trackers, and active block
    self.messages.append({})
    in_function_call = False
    self.active_block = None

    pprint("====== response ======")
    for chunk in response:
      delta = chunk["choices"][0]["delta"]
      self.messages[-1] = merge_deltas(self.messages[-1], delta)

      condition = "function_call" in self.messages[-1]
      if condition:
        if in_function_call == False:

          # If so, end the last block,
          self.end_active_block()

          # Print newline if it was just a code block or user message
          # (this just looks nice)
          last_role = self.messages[-2]["role"]
          if last_role == "user" or last_role == "function":
            print()

          # then create a new code block
          self.active_block = CodeBlock()

        # Remember we're in a function_call
        in_function_call = True

        # Now let's parse the function's arguments:

        # gpt-4
        # Parse arguments and save to parsed_arguments, under function_call
        if "arguments" in self.messages[-1]["function_call"]:
          arguments = self.messages[-1]["function_call"]["arguments"]
          new_parsed_arguments = parse_partial_json(arguments)
          if new_parsed_arguments:
            # Only overwrite what we have if it's not None (which means it failed to parse)
            self.messages[-1]["function_call"][
              "parsed_arguments"] = new_parsed_arguments


      else:
        in_function_call = False
        if self.active_block == None:
          self.active_block = MessageBlock()

      self.active_block.update_from_message(self.messages[-1])

      if chunk["choices"][0]["finish_reason"]:
        if chunk["choices"][0]["finish_reason"] != "function_call":
          self.active_block.end()
          return

        if chunk["choices"][0]["finish_reason"] == "function_call":
          # Time to call the function!
          # (Because this is Open Interpreter, we only have one function.)

          print("Running function:")
          print(self.messages[-1])
          print("---")

          # Ask for user confirmation to run code
          if self.auto_run == False:

            # End the active block so you can run input() below it
            # Save language and code so we can create a new block in a moment
            self.active_block.end()
            language = self.active_block.language
            code = self.active_block.code

            # Prompt user
            response = input("  Would you like to run this code? (y/n)\n\n  ")
            print("")  # <- Aesthetic choice

            if response.strip().lower() == "y":
              # Create a new, identical block where the code will actually be run
              self.active_block = CodeBlock()
              self.active_block.language = language
              self.active_block.code = code

            else:
              # User declined to run code.
              self.active_block.end()
              self.messages.append({
                "role":
                "function",
                "name":
                "run_code",
                "content":
                "User decided not to run this code."
              })
              return

          # If we couldn't parse its arguments, we need to try again.
          if not self.local and "parsed_arguments" not in self.messages[-1]["function_call"]:

            # After collecting some data via the below instruction to users,
            # This is the most common failure pattern: https://github.com/KillianLucas/open-interpreter/issues/41

            # print("> Function call could not be parsed.\n\nPlease open an issue on Github (openinterpreter.com, click Github) and paste the following:")
            # print("\n", self.messages[-1]["function_call"], "\n")
            # time.sleep(2)
            # print("Informing the language model and continuing...")

            # Since it can't really be fixed without something complex,
            # let's just berate the LLM then go around again.

            self.messages.append({
              "role": "function",
              "name": "run_code",
              "content": """Your function call could not be parsed. Please use ONLY the `run_code` function, which takes two parameters: `code` and `language`. Your response should be formatted as a JSON."""
            })

            self.respond()
            return

          # Create or retrieve a Code Interpreter for this language
          language = self.messages[-1]["function_call"]["parsed_arguments"][
            "language"]
          if language not in self.code_interpreters:
            self.code_interpreters[language] = CodeInterpreter(language, self.debug_mode)
          code_interpreter = self.code_interpreters[language]

          # Let this Code Interpreter control the active_block
          code_interpreter.active_block = self.active_block
          code_interpreter.run()

          # End the active_block
          self.active_block.end()

          # Append the output to messages
          # Explicitly tell it if there was no output (sometimes "" = hallucinates output)
          self.messages.append({
            "role": "function",
            "name": "run_code",
            "content": self.active_block.output if self.active_block.output else "No output"
          })

          # Go around again
          self.respond()