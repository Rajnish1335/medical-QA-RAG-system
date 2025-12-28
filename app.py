from flask import Flask, render_template, request
from src.helper import download_embedding
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt  # your updated system_prompt
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Embeddings + Vector Store
embeddings = download_embedding()
index_name = "medical-bot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# LLM
chatModel = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0
)

# Prompt template with chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human",
     "Conversation so far:\n{chat_history}\n\n"
     "User question: {question}\n"
     "Answer based on conversation and retrieved context.")
])

# Multi-user memory dictionary
user_memories = {}

def get_memory(user_id):
    """Return a ConversationBufferWindowMemory for the user."""
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=3 
        )
    return user_memories[user_id]

# Create conversational retrieval chain
def get_rag_chain(user_id):
    memory = get_memory(user_id)
    return ConversationalRetrievalChain.from_llm(
        llm=chatModel,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False
    )

# Routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    user_id = request.remote_addr  # simple unique user identifier
    print(f"User ({user_id}): {msg}")

    rag_chain = get_rag_chain(user_id)
    response = rag_chain.invoke({"question": msg})
    answer = response["answer"]

    print(f"Bot ({user_id}): {answer}")
    return answer

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
