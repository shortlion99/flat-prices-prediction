"""
Singapore Housing RAG Chatbot
A reusable RAG chatbot for Singapore housing information using LangChain and LangGraph.

Usage:
    from rag_chatbot import RAGChatbot
    
    # Initialize the chatbot
    bot = RAGChatbot()
    
    # Single question
    response = bot.chat("Tell me about Yishun")
    print(response)
    
    # Conversation with memory
    conversation = bot.start_conversation()
    response1 = bot.chat("Tell me about Yishun", conversation)
    response2 = bot.chat("What about housing prices there?", conversation)
"""

import os
import json
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# LangChain imports
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition


class RAGChatbot:
    """
    A RAG chatbot for Singapore housing information.
    
    This class encapsulates the entire RAG pipeline including:
    - Document loading and processing
    - Vector store creation
    - LangGraph workflow
    - Chat interface
    """
    # Data-path parameter required
    def __init__(self, data_file, env_file: str = ".env"):
        """
        Initialize the RAG chatbot.
        
        Args:
            data_file: Path(s) to JSON file(s) containing housing data. 
                      Can be a single string or list of strings for multiple files.
                      Example: "data.json" or ["data1.json", "data2.json"]
            env_file: Path to the .env file containing API keys
        """
        # Convert single file to list for uniform processing
        if isinstance(data_file, str):
            self.data_files = [data_file]
        elif isinstance(data_file, list):
            self.data_files = data_file
        else:
            raise ValueError("data_file must be a string or list of strings")
        
        self.env_file = env_file 
        
        # Load environment variables
        self._load_environment()
        
        # Initialize components
        self._initialize_models()
        self._create_vector_store()
        self._load_data()
        self._build_graph()
        
        print("✅ RAG Chatbot initialized successfully!")
    
    def _load_environment(self):
        """Load environment variables from .env file."""
        load_dotenv(self.env_file)
        
        # Verify required environment variables
        required_vars = ["MISTRAL_API_KEY"]
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Set environment variables
        os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")
        
        # Optional: LangSmith tracing
        if os.getenv("LANGSMITH_API_KEY"):
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    
    def _initialize_models(self):
        """Initialize the LLM and embeddings models."""
        try:
            self.llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
            self.embeddings = MistralAIEmbeddings(model="mistral-embed")
            print("✅ Models initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {e}")
    
    def _create_vector_store(self):
        """Create the vector store."""
        self.vector_store = InMemoryVectorStore(self.embeddings)
        print("✅ Vector store created")
    
    def _load_data(self):
        """Load and process the housing data from multiple files."""
        all_documents = []
        total_entries = 0
        
        for data_file in self.data_files:
            try:
                print(f"📁 Loading data from: {data_file}")
                
                # Load JSON data
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Convert to documents
                file_documents = []
                for entry in data:
                    area = entry["area"]
                    has_hdb = entry["has_hdb"]
                    hdb_price_range = entry["hdb_price_range"]
                    pros = entry["pros"]
                    cons = entry["cons"]
                    
                    doc = Document(
                        page_content=f"Area: {area}\nHDB Available: {has_hdb}\nHDB Price range: {hdb_price_range}\nPros: {pros}\nCons: {cons}",
                        metadata={"Area": area, "Source": data_file}
                    )
                    file_documents.append(doc)
                
                all_documents.extend(file_documents)
                total_entries += len(file_documents)
                print(f"  ✅ Loaded {len(file_documents)} entries from {data_file}")
                
            except FileNotFoundError:
                raise FileNotFoundError(f"Data file not found: {data_file}")
            except Exception as e:
                raise RuntimeError(f"Failed to load data from {data_file}: {e}")
        
        # Split all documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        json_splits = text_splitter.split_documents(all_documents)
        
        # Add to vector store
        self.vector_store.add_documents(documents=json_splits)
        
        print(f"✅ Successfully loaded {total_entries} total entries from {len(self.data_files)} file(s)")
        print(f"✅ Created {len(json_splits)} document chunks for vector store")
    
    # RAG pipeline =========================================
    def _build_graph(self):
        """Build the LangGraph workflow."""
        
        # Define the retrieve tool (needs to be instance method)
        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            retrieved_docs = self.vector_store.similarity_search(query, k=2)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\nContent: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs
        
        # Store the tool for later use
        self.retrieve_tool = retrieve
        
        # Define workflow functions
        def query_or_respond(state: MessagesState):
            """Generate tool call for retrieval or respond."""
            llm_with_tools = self.llm.bind_tools([self.retrieve_tool])
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        
        def generate(state: MessagesState):
            """Generate answer using retrieved content."""
            # Get recent tool messages
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]
            
            # Format into prompt
            docs_content = "\n\n".join(doc.content for doc in tool_messages)
            system_message_content = (
                "You are an assistant for question-answering tasks on the housing market in Singapore. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, say that you don't know. "
                "Use three sentences maximum and keep the answer concise.\n\n"
                f"{docs_content}"
            )
            
            conversation_messages = [
                message for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]
            
            prompt = [SystemMessage(system_message_content)] + conversation_messages
            response = self.llm.invoke(prompt)
            return {"messages": [response]}
        
        # Build the graph
        graph_builder = StateGraph(MessagesState)
        tools = ToolNode([self.retrieve_tool])
        
        graph_builder.add_node("query_or_respond", query_or_respond)
        graph_builder.add_node("tools", tools)
        graph_builder.add_node("generate", generate)
        
        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)
        
        self.graph = graph_builder.compile()
        print("✅ Graph workflow built successfully")
# ====================================================================
    
    def chat(self, message: str, conversation_state: Optional[Dict[str, Any]] = None) -> str:
        """
        Chat with the bot using a single message.
        
        Args:
            message: The user's message
            conversation_state: Optional conversation state for multi-turn conversations
            
        Returns:
            The bot's response as a string
        """
        if conversation_state is None:
            conversation_state = {"messages": []}
        
        # Add user message
        conversation_state["messages"].append(HumanMessage(content=message))
        
        # Get response from graph
        try:
            result = self.graph.invoke(conversation_state)
            response = result["messages"][-1].content
            
            # Update conversation state
            conversation_state.update(result)
            
            return response
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def start_conversation(self) -> Dict[str, Any]:
        """
        Start a new conversation.
        
        Returns:
            Empty conversation state
        """
        return {"messages": []}
    
    def chat_with_state(self, message: str, conversation_state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Chat with the bot and return both response and updated state.
        
        Args:
            message: The user's message
            conversation_state: Current conversation state
            
        Returns:
            Tuple of (response, updated_conversation_state)
        """
        response = self.chat(message, conversation_state)
        return response, conversation_state
    
    def get_conversation_history(self, conversation_state: Dict[str, Any]) -> list:
        """
        Get the conversation history.
        
        Args:
            conversation_state: Current conversation state
            
        Returns:
            List of messages in the conversation
        """
        return [
            {
                "type": msg.type,
                "content": msg.content,
            }
            for msg in conversation_state.get("messages", [])
            if msg.type in ("human", "ai")
        ]


# Convenience function for quick testing
def quick_chat(message: str) -> str:
    """
    Quick chat function for testing.
    
    Args:
        message: User message
        
    Returns:
        Bot response
    """
    bot = RAGChatbot()
    return bot.chat(message)


if __name__ == "__main__":
    # Example usage when run directly
    print("🤖 Singapore Housing RAG Chatbot")
    print("=" * 50)
    
    try:
        # Initialize chatbot
        bot = RAGChatbot()
        
        # Start conversation
        conversation = bot.start_conversation()
        
        print("\nBot: Hello! Ask me about Singapore housing areas. Type 'quit' to exit.\n")
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Bot: Goodbye!")
                break
            
            response = bot.chat(user_input, conversation)
            print(f"Bot: {response}\n")
            
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        print("Make sure you have:")
        print("1. A .env file with MISTRAL_API_KEY")
        print("2. The singapore_hdb_data.json file")
        print("3. All required packages installed")