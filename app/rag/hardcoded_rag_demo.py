from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
CHROMA_DIR = Path(__file__).parent / "chroma_wiki_class"

vectorstore = Chroma(
    collection_name="wiki_cricket_info",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR),
)


def retrieve_context(query: str) -> str:
    """Search cricket-related context from stored Wikipedia data."""
    results = vectorstore.similarity_search(query, k=3)
    return "\n\n".join(doc.page_content for doc in results)


llm = init_chat_model("gpt-4o", model_provider="openai")
agent = create_agent(llm, [retrieve_context])

question = "Who is Sachin Tendulkar?"

response = ""
for event in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    last_message = event["messages"][-1]
    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        response = last_message.content

print("Question:", question)
print("Answer:", response)
