from langchain_core.messages import HumanMessage
from agent import get_agent
from database import init_db

init_db()

agent = get_agent("gemini-3.8-flash")


config = {
    "configurable": {
        "thread_id": "test_thread_ml_blog",
    }
}

def chunk_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


for message_chunk, metadata in agent.stream(
    {"messages": [HumanMessage(content="Generate a blog about Machine learning")]},
    config=config,
    stream_mode="messages"
):
    text = chunk_text(message_chunk.content)
    if text:
        print(text, end="", flush=True)

print()