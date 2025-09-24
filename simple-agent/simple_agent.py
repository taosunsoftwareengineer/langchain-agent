import getpass
import os
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Get API keys
os.environ["LANGSMITH_TRACING"] = "true"
if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("Enter API key for LangSmith: ")
    print("LangSmith API key loaded successfully.")
    
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Enter API key for Tavily: ")
    print("Tavily API key loaded successfully.")
    
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google: ")
print("Google API key loaded successfully.")

# Define tools
search = TavilySearch(max_results=2)
tools = [search]

# for testing the search tool:
# search_results = search.invoke("What is the weather in SF")
# print(search_results)

# Using language models
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
model_with_tools = model.bind_tools(tools)

# Ask questions
query = "Search for the weather in SF"
response = model_with_tools.invoke([{"role": "user", "content": query}])

print(f"Message content: {response.text()}\n")
print(f"Tool calls: {response.tool_calls}")

# Create the agent
agent_executor = create_react_agent(model, tools)

# Run the agent
input_message = {"role": "user", "content": "Search for the weather in SF"}
response = agent_executor.invoke({"messages": [input_message]})

for message in response["messages"]:
    message.pretty_print()

    
# Adding in memory
memory = MemorySaver()
agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

for step in agent_executor.stream(
    {"messages": [("user", "Hi, I'm Tao!")]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
    
for step in agent_executor.stream(
    {"messages": [("user", "What is my name?")]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
