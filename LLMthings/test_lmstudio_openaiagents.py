import asyncio
import logging
from agents import Agent, Runner, trace
from openai import AsyncOpenAI
from agents import Agent, OpenAIResponsesModel, Runner, set_tracing_disabled, function_tool

set_tracing_disabled(True)
logging.basicConfig(level=logging.DEBUG)

#LMStudio
gpt_oss_model = OpenAIResponsesModel(
    model="openai/gpt-oss:20b",
    openai_client=AsyncOpenAI(
        base_url="http://192.168.15.9:1234/v1",
        api_key="lmstudio",
    ),
) 

@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny."

async def main():
    agent = Agent(
        name="Web searcher",
        instructions="You are a helpful agent.",
        tools=[get_weather],
        model = gpt_oss_model
    )

    with trace("Web search example"):
        result = await Runner.run(
            agent,
            "What's the weather in Tokyo?",
        )
        print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())