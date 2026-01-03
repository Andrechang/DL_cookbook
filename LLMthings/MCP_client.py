import asyncio
from fastmcp import Client

client = Client("http://127.0.0.1:8000/mcp")

async def main():
    async with client:
        # Basic server interaction
        await client.ping()
        
        # List available operations
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()
        
        print("Available tools:")
        for tool in tools:
            print(f"- {tool} \n\n")

        print("Available resources:", resources)
        print("Available prompts:", prompts)
        # Execute operations
        result = await client.call_tool("get_account_info", {"param": "value"})
        print(result)

asyncio.run(main())

