from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from rich import print

#1 creating tool

@tool
def get_text_length (text:str) -> int:
    """Returns the number of character in a given text"""
    return len(text)

import os
from dotenv import load_dotenv
load_dotenv()


model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key = os.getenv("GOOGLE_API_KEY")
)


#2 Tool binding iwth model;                         #bind all the tools that you have here!
model_with_tool=model.bind_tools([get_text_length]) #Now model has both the tool and query when invoked

#Important step: dict for all the tools
tools={
    'get_text_length' : get_text_length
}

#3 Tool Calling
message =[]
prompt = input("Enter the text you want to get length of: \n")
query = HumanMessage(f"Return the no. of characters in the given text: {prompt}")
message.append(query)

result=model_with_tool.invoke(message)
message.append(result)



if result.tool_calls:
    tool_name = result.tool_calls[0]["name"]
    ToolMessage = tools[tool_name].invoke(result.tool_calls[0])
    message.append(ToolMessage)

    result = model_with_tool.invoke(message)
    print(message)
    print(result.content)


#------------------------OUTPUT----------------------------

#Enter the text you want to get length of: 
# My Name is Nidhish
# [
#     HumanMessage(content='Return the no. of characters in the given text: My Name is Nidhish', additional_kwargs={}, response_metadata={}),
#     AIMessage(
#         content='',
#         additional_kwargs={
#             'function_call': {'name': 'get_text_length', 'arguments': '{"text": "My Name is Nidhish"}'},
#             '__gemini_function_call_thought_signatures__': {
#                 '1b43e319-9c5f-42ba-bb31-e840995b2d04': 
# 'CssCARFNMg8qhNL7Pn52guEyWqvxHmAym0aMPDnRKKzesstkz4acoY4pNUbAuTsKPIM2P1N6w/BIaDx1+LRZ//DJDcLvbEjuxaKV9ZVHfsnmUj+PBzK0yNNaO/K3VczSQH43b3iSxU+
# 6v3e3iL/B6MSBRQqBJp22vPlqvzisn3GFF+WFchPjV9sSo2lxN1JV+ouJq6n3qNpQ1k8fzqVVSuxpvPNnDYNpsz1cyKsrv0eGOvtQJAZZkHn36baFap1WdlJi+6PL2cdETgq+RFVqvAd
# LGVzLJ6L+32cbC7+lJlk81ra4uhfDh39OKqpIorp5gADF1EJ1pMcdxXEWoKAse4ZWyCNRCXoamfey97N93figpEb3oqkNH0i4ksBKrHj7FVNojyJRfbDhs5jwWxP4sFGYW4CxVZ00/nz
# vBl3SsE437fT54mRcyJfeGET9qA=='
#             }
#         },
#         response_metadata={
#             'finish_reason': 'STOP',
#             'model_name': 'gemini-2.5-flash',
#             'safety_ratings': [],
#             'model_provider': 'google_genai'
#         },
#         id='lc_run--019fe05b-dbd2-7ee1-b896-e40bef849b55-0',
#         tool_calls=[
#             {
#                 'name': 'get_text_length',
#                 'args': {'text': 'My Name is Nidhish'},
#                 'id': '1b43e319-9c5f-42ba-bb31-e840995b2d04',
#                 'type': 'tool_call'
#             }
#         ],
#         invalid_tool_calls=[],
#         usage_metadata={
#             'input_tokens': 61,
#             'output_tokens': 96,
#             'total_tokens': 157,
#             'input_token_details': {'cache_read': 0},
#             'output_token_details': {'reasoning': 74}
#         }
#     ),
#     ToolMessage(content='18', name='get_text_length', tool_call_id='1b43e319-9c5f-42ba-bb31-e840995b2d04')
# ]


# The number of characters in the given text is 18.

