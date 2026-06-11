from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# This creates an embedding model. An embedding converts texts into numbers. Those numbers capture meaning.
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vectorstore = Chroma(
    collection_name="wiki_cricket_info",
    embedding_function=embeddings,
    persist_directory="./chroma_wiki_class",
)


def retrieve_context(query: str) -> str:
    """Search cricket-related context from stored Wikipedia data."""
    results = vectorstore.similarity_search(query, k=3)
    return "\n\n".join(doc.page_content for doc in results)


llm = init_chat_model("gpt-4o", model_provider="openai")
agent = create_agent(llm, [retrieve_context])

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    response = ""
    for event in agent.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        stream_mode="values",
    ):
        last_message = event["messages"][-1]
        if hasattr(last_message, "content") and isinstance(last_message.content, str):
            response = last_message.content

    print("Bot:", response)
