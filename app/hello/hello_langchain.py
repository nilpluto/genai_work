
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
# Load environment variables from .env file instead of hardcoding them in the cod
load_dotenv()


# model = init_chat_model("gpt-4o-mini", model_provider="openai")
# model = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
model = init_chat_model("google_genai:gemini-2.5-flash-lite")
response = model.invoke("who is Dhoni")
print(response.content)
