# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# import asyncio
# import os

# # Load environment variables
# load_dotenv()

# async def main():
#     # Initialize MCP client
#     client = MultiServerMCPClient(
#         {
#             "assistant": {
#                 "command": "python",
#                 "args": ["./AssistantServer.py"],  # Path relative to project root
#                 "transport": "stdio",
#             }
#         }
#     )

#     # Set Groq API key
#     os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

#     # Get tools and initialize agent
#     tools = await client.get_tools()
#     model = ChatGroq(model="llama-3.1-8b-instant")
#     agent = create_react_agent(model, tools)

#     # Test queries
#     queries = [
#         {"messages": [{"role": "user", "content": "Check important emails for user abhiram"}]},
#         {"messages": [{"role": "user", "content": "Send an email to abhiramtemp@gmail.com with subject 'Test' and body 'Hello from MCP!'"}]},
#         {"messages": [{"role": "user", "content": "Add a reminder for abhiram to buy milk due tomorrow"}]},
#         {"messages": [{"role": "user", "content": "Delete task buy milk for abhiram"}]},
#         {"messages": [{"role": "user", "content": "Google search Python tutorials"}]},
#         {"messages": [{"role": "user", "content": "What is LlamaIndex?"}]},
#         {"messages": [{"role": "user", "content": "Schedule a meeting with ram@example.com at 5:00 PM"}]},
#         {"messages": [{"role": "user", "content": "Summarize tasks for abhiram"}]},
#         {"messages": [{"role": "user", "content": "Query database: select * from tasks where user_id = 'abhiram'"}]}
#     ]

#     for query in queries:
#         response = await agent.ainvoke(query)
#         print(f"Query: {query['messages'][0]['content']}")
#         print(f"Response: {response['messages'][-1].content}\n")

# if __name__ == "__main__":
#     asyncio.run(main())


from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import asyncio
import os

# Load environment variables
load_dotenv()

async def main():
    # Initialize MCP client
    client = MultiServerMCPClient(
        {
            "assistant": {
                "command": "python",
                "args": ["./AssistantServer.py"],  # Path relative to project root
                "transport": "stdio",
            }
        }
    )

    # Set Groq API key
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    # Get tools and initialize agent
    tools = await client.get_tools()
    model = ChatGroq(model="llama-3.1-8b-instant")
    agent = create_react_agent(model, tools)

    # Test queries
    queries = [
        {"messages": [{"role": "user", "content": "Check important emails for user abhiram"}]},
        {"messages": [{"role": "user", "content": "Send an email to abhiramtemp@gmail.com with subject 'Test' and body 'Hello from MCP!'"}]},
        {"messages": [{"role": "user", "content": "Add a reminder for abhiram to buy milk due tomorrow"}]},
        {"messages": [{"role": "user", "content": "Delete task buy milk for abhiram"}]},
        {"messages": [{"role": "user", "content": "Google search Python tutorials"}]},
        {"messages": [{"role": "user", "content": "What is LlamaIndex?"}]},
        {"messages": [{"role": "user", "content": "Schedule a meeting with ram@example.com at 5:00 PM"}]},
    ]

    for query in queries:
        response = await agent.ainvoke(query)
        print(f"Query: {query['messages'][0]['content']}")
        print(f"Response: {response['messages'][-1].content}\n")

if __name__ == "__main__":
    asyncio.run(main())