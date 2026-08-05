from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

data = TextLoader("doc_chunking/text.txt")
doc = data.load()



#1. Character based chunking: 

splitter  = CharacterTextSplitter(
    separator="",#if not mentioned that is, by default splitter is only considered by \n\n
    chunk_size=10,
    chunk_overlap=1
)

chunks = splitter.split_documents(doc)

print("Chunked document via CHARACTER BASED SPLITTER into : ",len(chunks)," chunks")
print("Chunks are: \n")
for i in chunks:
    print(i.page_content)
print("\n\n\n")


#2. Token based chunking : 

from langchain_text_splitters import TokenTextSplitter


splitter  = TokenTextSplitter(
    #each token is counted and then seperated on the basis of size
    chunk_size=10,
    chunk_overlap=1
)

chunks = splitter.split_documents(doc)

print("Chunked document via TOKEN BASED SPLITTER into : ",len(chunks)," chunks")
print("Chunks are: \n")
for i in chunks:
    print(i.page_content)
print("\n\n\n")

#3. Recursive character based chunking: 

from langchain_text_splitters import RecursiveCharacterTextSplitter


splitter  = RecursiveCharacterTextSplitter(
    #automatically splits on the basis in order of ["\n\n","\n"," ",""]
    chunk_size=10,
    chunk_overlap=1
)

chunks = splitter.split_documents(doc)

print("Chunked document via RECURSIVE SPLITTER into : ",len(chunks)," chunks")
print("Chunks are: \n")
for i in chunks:
    print(i.page_content)
print("\n\n\n")
