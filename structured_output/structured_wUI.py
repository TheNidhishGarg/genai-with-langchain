

# Strucutred ouptut using model.with_structured_output().


import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

class Movie(BaseModel):
    title: str
    release_yr: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

structured_model = model.with_structured_output(Movie)

st.title("Movie Summarizer")
st.subheader("Summarise your movie's detailed summary into finest and particular details!")

para = st.text_area("Insert movie paragraph:")

if st.button("Summarize"):
    movie_data = structured_model.invoke(para)

    st.subheader("Structured Output")
    st.write(movie_data)

    print(movie_data)
