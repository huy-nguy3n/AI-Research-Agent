from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor

from tools import search_tool, wikipedia_tool, save_to_txt_tool

# Load .env (e.g., API keys)
load_dotenv()

# Define structured output
class ResearchResponseModel(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Language model
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
# llm2 = ChatDeepSeek(model="deepseek-chat")
# llm3 = ChatOpenAI(model="gpt-4")

# Output parser
parser = PydanticOutputParser(pydantic_object=ResearchResponseModel)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant. Your job is to help me find information on a specific topic.
            Answer the user query and use the necessary tools to gather information.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Tools
tools = [search_tool, wikipedia_tool, save_to_txt_tool]

# Agent setup
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run loop
continue_research = True

while continue_research:
    query = input("\nWhat can I help you research today?: ")
    raw_response = agent_executor.invoke({"query": query})

    try:
        structured_response = parser.parse(raw_response.get("output")[0]["text"])

        # Format data as string
        data_str = (
            f"Topic: {structured_response.topic}\n\n"
            f"Summary:\n{structured_response.summary}\n\n"
            f"Sources:\n" + "\n".join(f"- {s}" for s in structured_response.sources) + "\n\n"
            f"Tools Used:\n" + "\n".join(f"- {t}" for t in structured_response.tools_used)
        )

        print("\n" + data_str)

    except Exception as e:
        print("Error parsing the response:", e)
        print("Raw response:", raw_response)
        structured_response = None

    # Ask to save
    save_decision = input("\nWould you like to save this research to a file? (y/n): ").strip().lower()
    if save_decision == "y" and structured_response:
        save_to_txt_tool.invoke({
            "data": data_str,
            "filename": "research_response.txt"
        })
        print("Research saved.")

    # Ask to continue
    continue_decision = input("\nDo you want to continue researching? (y/n): ").strip().lower()
    if continue_decision != "y":
        continue_research = False
        print("Thanks for using the research assistant.")
