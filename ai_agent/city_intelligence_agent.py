# This Agent extracts City info by providing the WEATHER & NEWS of that city.
# built using OpenWeather API and Tavily Api.


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from langchain_core.tools import tool
import requests
from rich import print
from tavily import TavilyClient

import os
from dotenv import load_dotenv
load_dotenv()


model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key = os.getenv("GOOGLE_API_KEY")
)

#weather Tool

@tool
def get_weather(city: str) -> str:
    """Get weather of the city"""
    api_key = os.getenv("WEATHER_API_KEY")
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"

    response = requests.get(url)
    data = response.json()

    if str(data.get("cod"))!= "200":
        return f"Error: {data.get('message','Could not fetch weather')}" 

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {city}: {desc}, {temp}C"


# print(get_weather.invoke("haryana"),"\n\n")



# news tool

tavily_client = TavilyClient(api_key = "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")

@tool
def get_news(city:str) -> str:
    """Get latest news about the city. always return in the format of- Title: Url: News:"""

    response = tavily_client.search(
        query = f"latest news in {city} in English",
        search_depth="basic",
        max_results=3
    )

    results=response.get("results",[])

    if not results:
        return f"No news found for {city}"
    
    news_list = []

    for i in results:
        title = i["title"]
        url=i["url"]
        content = i["content"]
        prompt = f"Just return the direct summary of the news content of this in 3-4 lines nothing: {content}"
        content_summary=model.invoke(prompt)

        news_list.append(
            f"{title} \n url - {url} \n\n {content_summary.content} \n\n\n "
        )

    return f"Lates news in {city} : \n\n"+ "\n\n".join(news_list)

# print(get_news.invoke("Haryana"))


#Tool Binding
tools = {
    "get_weather" : get_weather,
    "get_news" : get_news
}

model_with_tool = model.bind_tools([get_weather,get_news])



#AgentLOOP
messages = []

print("City intelligence system")
print("Type Exit to quit")

while True:
    user_input = input("You: ")
    if user_input.lower()=="exit":
        break
    messages.append(HumanMessage(content=user_input))

    while True:
        result = model_with_tool.invoke(messages)
        messages.append(result)
        #this will return tool required data

        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call['name']

                #HUMAN IN THE LOOP
                confirm = input(f"Agent wants to call {tool_name}.\nApprove (y/n): ")

                if confirm.lower() == "no":
                    print("Tool access denied, can not get information. Exiting ...")
                    break

                tool_result = tools[tool_name].invoke(tool_call)

                messages.append(
                    ToolMessage(
                        content=tool_result,
                        tool_call_id = tool_call['id']
                    )
                )

            continue
        else:
            print(result.content)
            break