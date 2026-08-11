from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()


model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key = os.getenv("GOOGLE_API_KEY")
)

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant
summarise the following news into clear bullet points

{news}
"""
)

parser = StrOutputParser()

chain = prompt | model | parser

# BASIC DONE ************************************************************


#getting data from tavily search
search_tool = TavilySearchResults(max_results =3,tavily_api_key = "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")

#giving data to model and printing it!
news_result = search_tool.run("Latest AI news of August 2026")
result = chain.invoke({"news" : news_result})
print(result)



# ----------------------------------------OUTPUT--------------------------------------------
# Here's a summary of the AI news for August 2026:

# **Key Market Trends & Developments:**
# *   **Price Cuts & Competition:** AI model pricing saw significant reductions, with OpenAI cutting GPT-5.6 Luna's price by 80% to $0.20 per million input tokens. This contributes to an accelerating "speed race, pricing war, and distribution war" in the AI market, with major model launches quadrupling since 2023.
# *   **Agent Rollout & Integration:** AI agents are being more widely rolled out and deeply integrated into daily applications, such as Google Gemini replacing Google Assistant on Android.
# *   **Regulatory Scrutiny:** Major AI releases are facing increased government review and tighter U.S. regulations.
# *   **User Growth:** ChatGPT reached approximately 1 billion weekly active users.

# **New AI Models & Tools Released:**
# *   **Prime Intellect's Prime Agent:** An open-source coding agent that scored 95.5% on ARC-AGI-3, designed to improve AI models' performance on complex tasks using a persistent Python environment.
# *   **Cursor’s Mixture-of-Kittens (MoK):** An open-source optimized Mixture-of-Experts (MoE) megakernel that boosts efficiency and performance on NVL72s GPUs.
# *   **DiffusionGemma:** A discrete diffusion model adapted from Gemma 4, capable of refining 256-token blocks in parallel and achieving high output speeds (around 1,500 tokens/second on an NVIDIA H100).
# *   **NVIDIA's Alpamayo 2 Super:** A commercially licensed reasoning model for robotaxis and autonomous vehicles, designed for rare driving scenarios with inspectable decisions and broad multitask capabilities.
# *   **New Productivity Agents:** Google released Gemini Spark ($99.99/month, cloud-based) and Anthropic launched Claude Cowork ($20/month, desktop-first) as "always-on" agents for office and knowledge workers.
# *   **Other Model Releases:** The market also saw releases or increased visibility for models like DeepSeek-V4-Flash-0731, GPT-5.6 Luna, Meta Muse Spark 1.1, and Thinking Machines Inkling.

# **Company News:**
# *   **Google DeepMind Leadership Changes:** Demis Hassabis transitioned to Chair of Google DeepMind and Chief Scientist of Alphabet, while Jeff Dean departed after 27 years to launch Discovery Loop. This announcement led to a more than 5% fall in Alphabet shares.
