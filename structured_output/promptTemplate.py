import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    ('system',
     """
You are an expert movie summarizer.

Your task is to convert the given movie or TV series description into concise bullet points.

Instructions:
- Extract only the most important information.
- Use the fewest words possible.
- Cover:
  • Main plot
  • Main characters
  • Major events
  • Key twists (if present)
  • Important locations
  • Themes
  • Ending (only if mentioned)
- Remove filler, opinions, and unnecessary adjectives.
- Each bullet should contain only one key point.
- Maximum 8–15 words per bullet.
- Do NOT rewrite into paragraphs.
- Do NOT explain.
- Do NOT add information not present in the input.
- Preserve chronological order whenever possible.

Output Format:

• ...
• ...
• ...
• ...
"""),
    ('human',
     """
Summarize the following movie/TV series description.

Description:
{movie_description}
""")
])

st.title("🎬 Movie Summarizer")

para = st.text_area("Insert movie paragraph:")

if st.button("Summarize"):
    final_prompt = prompt.invoke({"movie_description": para})
    response = model.invoke(final_prompt)
    st.write(response.content)