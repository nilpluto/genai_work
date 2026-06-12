import os
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from dotenv import load_dotenv
load_dotenv()


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


async def run_agent():

    client = MultiServerMCPClient(  # it can connect to multiple MCP server
        {
            "NilanjanFileSystem": {
                "command": "python",  # run python package
                "args": [
                    "./filesystem_mcp.py"  # file that contains the tool implementation,
                ],
                "transport": "stdio"  # communication channnel between agent and tool server
            }
        }
    )

    tools = await client.get_tools()
    agent = create_agent("google_genai:gemini-2.5-flash-lite", tools)

    response = await agent.ainvoke({"messages": "create a file named arun.txt"})
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(run_agent())
