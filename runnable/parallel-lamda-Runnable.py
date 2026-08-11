from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda



#defining model
import os
from dotenv import load_dotenv
load_dotenv()


model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key = os.getenv("GOOGLE_API_KEY")
)

#Two Prompts: 

# prompt1
short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

# prompt 2
long_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail but not more than 300 words"
)

# input 
topic = "GenerativeAI"

# parser 
parser = StrOutputParser()

# making parallel runnable if the TemplateVariable is same
#*****************************************************************************
# chain=RunnableParallel({
#     "short" : short_prompt | model | parser,
#     "long" : long_prompt | model | parser
# })
#*****************************************************************************

# making paralle runnnable if each prompt has its own variable 
chain = RunnableParallel({
    "short" : RunnableLambda(lambda x: x["short"]) | short_prompt | model | parser,
    "long" : RunnableLambda(lambda x:x["long"]) | long_prompt | model |parser
})

# if the var is same 
#******************************************************************
# result = chain.invoke({"topic":topic})                          |
#******************************************************************

#if var for both the prompt template are different
result = chain.invoke({
    "short": {"topic" : "Machine Learning"},
    "long" : {"topic": "Generative AI"}
})

print("SHORT ANSWER : \n",result["short"])
print("LONG ANSWER : \n",result["long"])