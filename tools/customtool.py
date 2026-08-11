from langchain.tools import tool

@tool
def get_greetings(name:str)-> str: #type hints
    """Generate greeting message for a user""" # doc string

    return f"Hello {name},welcome to AI world"


print(get_greetings.invoke({"name": "Nidhish"}))

print(get_greetings.name)
print(get_greetings.description)
print(get_greetings.args)


