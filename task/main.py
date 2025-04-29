# Implement the Python code using the `openai` library to create the assistant.
# Start with a system prompt for the chat completion API.
# Send a static message and receive a response from the API.
# Show the chat completion message.
# from pyexpat.errors import messages
from calendar import firstweekday

import openai
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
import uuid


def end_conversation():
    return {"call_id": f"call_{uuid.uuid4().hex[:24]}"}


def main():

    #loading env file
    load_dotenv("key.env")
    api_key = os.environ.get("API_KEY", None)

    # initialize openAI API
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://litellm.aks-hs-prod.int.hyperskill.org/",
    )

    functions_list = [
        {
            "type": "function",
            "function": {
                "name": "end_conversation",
                "description": "Ends the conversation and returns a unique call ID.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "assistant",
                "content": "You are an helpful assistant for a simple CLI chat. Only respond with text messages. Get creative with the answers!"
            }
        ]
    )

    # calculate token usage
    MODEL_35_TURBO = "gpt-3.5-turbo"
    MODEL_4_TURBO = "gpt-4-turbo-preview"

    MODELS = {
        MODEL_35_TURBO: {"input_cost": 0.0005 / 1000, "output_cost": 0.0015 / 1000},
        MODEL_4_TURBO: {"input_cost": 0.01 / 1000, "output_cost": 0.03 / 1000},
    }

    def calculate_tokens_cost(model, chat_completion):
        if model not in MODELS:
            raise ValueError(f"Model {model} is not supported.")

        model_costs = MODELS[model]
        input_tokens_cost = chat_completion.usage.prompt_tokens * model_costs["input_cost"]
        output_tokens_cost = (
                chat_completion.usage.completion_tokens * model_costs["output_cost"]
        )
        return input_tokens_cost + output_tokens_cost

    # function to get API response
    def get_chat_completion_json(messages, tools = None, tool_choice="auto"):
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0.7,
            response_format={"type": "json_object"},
        )

    # chat loop
    while True:
        user_input = input("Enter a message: ").strip()
        messages = [{"role": "user", "content": f"{user_input}. You are an assistant that responds in structured JSON format. Answer to prompts correctly"}]

        chat_completion_json = get_chat_completion_json(messages, tools=functions_list, tool_choice="auto")

        # Check if OpenAI called our end_conversation function
        tool_calls = chat_completion_json.choices[0].message.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                if user_input.lower() == "end conversation":
                    call_result = end_conversation()
                    call_id = call_result["call_id"]
                    print(call_id)
                    print(f"""
                            You: {user_input}
                            Assistant: None
                            Cost: ${calculate_tokens_cost(MODEL_35_TURBO, chat_completion_json):.8f}
                        """)
                    return  # Exit loop
            # Process regular chatbot response
        gpt_response_json = chat_completion_json.choices[0].message.content
        response_data = json.loads(gpt_response_json)
        assistant_response = next(iter(response_data.values()), "Unknown response")

        print(f"""
               You: {user_input}
               Assistant: {assistant_response}
               Cost: ${calculate_tokens_cost(MODEL_35_TURBO, chat_completion_json):.8f}
           """)

    # print(res["response"]["description"])

    # print(")




if __name__ == "__main__":
    main()



