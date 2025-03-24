from opto import trace
from opto.trace import node, bundle
from opto.optimizers import OptoPrime
import ollama
import datetime
import re

# Select the model
# Llama2 cannot call tools
MODEL = "llama2:7b"
# Llama3 can call tools
# MODEL = "llama3.1:8b"

TEMPERATURE = 0.7


def get_date_feedback(predict):
    date_regex = r"^\d{4}-\d{2}-\d{2}$"
    # Validate that the input is in the correct format.
    if not re.fullmatch(date_regex, predict):
        return "Error: Date format incorrect. Expected YYYY-MM-DD."
    
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    if predict == date:
        return "Success!"
    else:
        # Note: this is not behavior cloning, but RL. For BC also return correct date (but that might lead to overfitting.)
        return f"Error: Wrong date."
    
       
    
@trace.model
class LLMAgent:

    def get_current_time(args=None):
        """Returns the current time."""
        now = datetime.datetime.now()
        return {
            "current_time": now.strftime("%H:%M:%S"),
            "current_date": now.strftime("%Y-%m-%d"),
            "timezone": datetime.datetime.now().astimezone().tzname() or "UTC"
        }

    def evaluate_arithmetic(args):
        """
        Evaluate a numeric arithmetic expression.
        The caret '^' is replaced with Python's exponentiation operator '**'.
        
        Args:
            args: Dictionary containing the expression to evaluate
        """
        expression = args.get("expression", "")
        safe_expression = expression.replace('^', '**')
        try:
            result = eval(safe_expression, {"__builtins__": {}})
            return result
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {e}")

    # Define the tools
    TOOLS = {
        "evaluate_arithmetic": {
            "definition": {
                "type": "function",
                "function": {
                    "name": "evaluate_arithmetic",
                    "description": "Evaluate a simple arithmetic expression provided as input. The expression should be numeric only and may contain +, -, *, /, parentheses, and '^' for exponentiation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string", 
                                "description": "The arithmetic expression to evaluate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            },
            "handler": evaluate_arithmetic
        },
        
        "get_current_time": {
            "definition": {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time, date, and timezone information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "format": {
                                "type": "string",
                                "description": "Optional format for the time (default, 24h, 12h)",
                                "enum": ["default", "24h", "12h"]
                            }
                        },
                        "required": []
                    }
                }
            },
            "handler": get_current_time
        }
    }

    def __init__(self, model: str):
        """
        Initialize the agent with the Ollama model and enabled tools.
        
        Args:
            model: The name of the Ollama model to use
            enabled_tools: List of tool names to enable (None means all tools)
        """
        self.model = model
    
    def get_tool_definitions(self):
        """Get the definitions of all enabled tools."""
        return [LLMAgent.TOOLS[name]["definition"] for name in LLMAgent.TOOLS]

    # def execute_tool(self, tool_name, arguments):
    #     """Execute a tool by name with the given arguments."""
    #     tool_name_str = tool_name
    #     return LLMAgent.TOOLS[tool_name_str]["handler"](arguments)
    #     # if tool_name in LLMAgent.TOOLS:
    #     #     return LLMAgent.TOOLS[tool_name]["handler"](arguments)        
    #     # raise ValueError(f'Tool "{tool_name}" not available')


    @bundle(trainable=True)
    def process_query(self, user_query: str) -> str:
        """
        Process the user query:
          1. Send the query to the LLM via Ollama.
          2. If the LLM requests a tool call, execute the function.
          3. Send the function result back to the LLM.
          4. Return the LLM's final answer.
        """
        # system_msg = node("Call tools only when necessary. If you can answer the question directly, do so. When you receive the answer from a tool, use the answer as directly as possible.",
        #                   trainable=True)

        # Send the initial query to the LLM.
        tools = self.get_tool_definitions()
        messages = [{"role": "user", "content": user_query}]

        print("PROMPT >>>", messages)
        initial_response = ollama.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            options={"temperature": TEMPERATURE},
        )
        print("RESPONSE >>>", initial_response)
 
        message = initial_response["message"]

        # Check if there are any tool calls
        tool_calls = message["tool_calls"]

        if tool_calls:
            for tool_call in tool_calls:
                if "function" in tool_call:
                    tool_name = tool_call["function"]["name"]
                    arguments = tool_call["function"]["arguments"]
                    
                    # Execute the appropriate tool
                    try:
                        tool_result = LLMAgent.TOOLS[tool_name]["handler"](arguments)
                        # tool_result = self.execute_tool(tool_name, arguments)
                        messages = [{"role": "user", "content": user_query},
                                    message,
                                    {"role": "tool", "name": tool_name, "content": str(tool_result)}]
                        print("PROMPT >>>", messages)
                        followup_response = ollama.chat(
                            model=self.model,
                            messages=messages,
                            tools=None,
                            options={"temperature": TEMPERATURE},
                        )
                        print("RESPONSE >>>", followup_response)

                        # Thsi will fire and return for the first tool that executed successfully.
                        return followup_response["message"]["content"]
                    
                    except ValueError as e:
                        print(f"Tool execution error: {e}")
        
        # If no function call is requested, assume the LLM returned a direct final answer.
        return message["content"]
        

    # @bundle(trainable=True)
    # def _call_ollama(self,
    #                 #  system_msg: str,
    #                  messages: list,
    #                  tools: list = None) -> dict:
    #     """
    #     Call the LLM with the user query.
    #     """
    #     verbose = True
    #     # messages = [{"role": "system", "content": system_msg}] + messages
    #     if verbose: print(f"QUERY: {messages}")
    #     response = ollama.chat(
    #         model=self.model,
    #         messages=node_to_plain(messages),
    #         tools=node_to_plain(tools),
    #         options={"temperature": 0.7},
    #     )
    #     if verbose: print(f"RESPONSE: {response}\n")
    #     return response

    # def _call_ollama_initial(self,
    #                         #  system_msg: str,
    #                          user_query: str) -> dict:
    #     """
    #     Call the LLM with the initial query. The payload includes the tools definitions so that
    #     the LLM can decide if a tool call is needed.
    #     """
    #     # Get all enabled tool definitions
    #     tools = self.get_tool_definitions()
        
    #     # Call the model using the proper format for tools
    #     messages = [
    #       {"role": "user", "content": user_query}
    #     ]
    #     return self._call_ollama(
    #         # system_msg,
    #         messages,
    #         tools
    #     )
        
    # def _call_ollama_followup(self,
    #                         #   system_msg: str,
    #                           user_query: str,
    #                           assistant_message: dict,
    #                           tool_name: str,
    #                           tool_result) -> dict:
    #     """
    #     Send the function result back to the LLM as a follow-up message.
    #     This message includes the original user query, the actual assistant message with tool calls,
    #     and the function result.
    #     """
    #     messages = [
    #         {"role": "user", "content": user_query},
    #         assistant_message,
    #         {"role": "tool", "name": tool_name, "content": str(tool_result)}
    #     ]
        
    #     return self._call_ollama(
    #         # system_msg,
    #         messages
    #     )


def test_tool_execution():
    agent = LLMAgent(MODEL)
    
    # Test using the time tool
    query2 = "What day is today? Return it in the format YEAR-MONTH-DAY, without any additional text."
    # query2 = "What is love?"
    print("\nQuery:", query2)
    answer2 = agent.process_query(query2)
    print("===============\nResponse:", answer2.data)
    
    # # Test case 1: Query with arithmetic expression.
    # query1 = "What is 1+3/(25+7)^2?"
    # print("Query:", query1)
    # answer1 = agent.process_query(query1)
    # print("===============\nResponse:", answer1.data)



# Example usage and tests
if __name__ == "__main__":
  test_tool_execution()
