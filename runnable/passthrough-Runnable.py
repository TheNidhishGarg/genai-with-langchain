from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough,RunnableParallel,RunnableLambda

#defining model
import os
from dotenv import load_dotenv
load_dotenv()


model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key = os.getenv("GOOGLE_API_KEY")
)

#parser
parser = StrOutputParser()

#prompts
code_gen = ChatPromptTemplate.from_messages({
    ("system","""
you are a genuine code generator    
"""),("human","""
{topic}
""")
})

code_explain = ChatPromptTemplate.from_messages({
    ("system","""
Explain the given code in 100words.
"""),("human","""
{code}
""")
})

#*********************************************************************************************************

#runnable sequence making
chain = code_gen | model | parser | code_explain | model | parser

#assigning the variable for the template
print("******WITHOUT PASSTHROUGH RUNNABLE******")
result = chain.invoke({"topic":"write a small code for palindrome"})
print(result,"\n\n\n")

# the issue here is, it will print the explanation, but NOT THE CODE as it is between the sequence/ chain
# if we want to use the output coming from middle of the chain we need to use output parser!

#*************************************************************************************************************


#passthrough runnable: 
seq = code_gen | model | parser

seq2 = RunnableParallel({
    "code": RunnablePassthrough(),
    "explanation" : code_explain | model | parser
})

final_seq = seq | seq2


print("******PASSTHROUGH RUNNABLE******")
result = final_seq.invoke({"topic":"Create code of plaindrome check"})
print(result["code"],"\n\n")
print(result["explanation"],"\n\n\n")