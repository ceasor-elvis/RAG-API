from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import MemorySaver
from langchain import hub
from schemas import State
from langgraph.graph import START, StateGraph

import os
from dotenv import load_dotenv

load_dotenv()

class Rag:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("LLM_API_KEY"), temperature=0.3)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("LLM_API_KEY"))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap=200,
            add_start_index=True
        )
        self.vector_store = Chroma(embedding_function=self.embeddings)
        self.prompt = hub.pull("rlm/rag-prompt")
        self.memory = MemorySaver()
        self.workflow = StateGraph(State)

    def split_into_embeddings(self, docs: str):
        if isinstance(docs, str):
            splits = self.text_splitter.split_text(docs)
        elif isinstance(docs, list):
            splits = self.text_splitter.split_documents(docs)
        else:
            raise ValueError("Input must be either a string or a list of strings.")
        return splits
    
    def store_documents(self, docs: str) -> list:
        """Returns a list of document ids"""
        all_splits = self.split_into_embeddings(docs)
        
        if isinstance(docs, str):
            ids = self.vector_store.add_texts(all_splits)
        elif isinstance(docs, list):
            ids = self.vector_store.add_documents(all_splits)
        return ids

    def retrieve(self, state:State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response}

    def response(self, thread_id, query):
        # Define the (single) node in the graph
        graph_builder = self.workflow.add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        self.workflow.add_node("retrieve", self.generate)

        app = self.workflow.compile(checkpointer=self.memory)
        config = {"configurable": {"thread_id": thread_id}}

        return app.invoke({"question": query}, config)   