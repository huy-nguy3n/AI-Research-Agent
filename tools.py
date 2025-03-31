from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool, StructuredTool
from datetime import datetime

# Search using DuckDuckGo (FREE)
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Useful for searching the web for information."
)

# Wikipedia
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Save to a text file
def save_to_txt(data: str, filename: str = "research_response.txt"):
    time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    text = (
        f"--- Research Response ---\n"
        f"Date: {time}\n\n"
        f"{data}\n"
        f"{'-' * 30}\n\n"
    )
    with open(filename, "a") as file:
        file.write(text)
    return f"Data saved to {filename} successfully."

save_to_txt_tool = StructuredTool.from_function(
    func=save_to_txt,
    name="save_text_file",
    description="Saves the research response to a text file. Requires 'data' and optional 'filename'."
)
