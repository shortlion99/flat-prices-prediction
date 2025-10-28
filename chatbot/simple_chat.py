"""
Simple Chat Interface for RAG Chatbot
Interactive script to chat with your Singapore Housing RAG bot
"""

import os
from rag_chatbot import RAGChatbot

def main():
    print("🏠 Singapore Housing RAG Chatbot")
    print("=" * 50)
    
    try:
        # Initialize bot with your datasets
        print("📂 Loading datasets...")
        data_dir = './Datasets'
        data_files = [os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('.json')
        ]
        
        print(f"📊 Found {len(data_files)} data files:")
        for file in data_files:
            print(f"   - {file}")
        
        print("\n🤖 Initializing chatbot...")
        bot = RAGChatbot(data_file=data_files)
        
        # Start conversation
        conversation = bot.start_conversation()
        
        print("\n✅ Bot ready! Ask me about Singapore housing.")
        print("💡 Try questions like:")
        print("   - 'Tell me about Yishun'")
        print("   - 'What areas have affordable HDB flats?'")
        print("   - 'Compare Tampines and Jurong'")
        print("   - 'Which areas are good for families?'")
        print("\n📝 Commands:")
        print("   - Type 'quit' or 'exit' to end")
        print("   - Type 'clear' to start a new conversation")
        print("   - Type 'help' to see this message again")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🙋 You: ").strip()
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                    print("👋 Thanks for chatting! Goodbye!")
                    break
                
                elif user_input.lower() in ['clear', 'reset', 'new']:
                    conversation = bot.start_conversation()
                    print("🔄 Started new conversation!")
                    continue
                
                elif user_input.lower() in ['help', 'h', '?']:
                    print("\n💡 Try questions like:")
                    print("   - 'Tell me about [area name]'")
                    print("   - 'What areas have affordable housing?'")
                    print("   - 'Compare [area1] and [area2]'") 
                    print("   - 'Which areas are good for families?'")
                    print("   - 'What are the pros and cons of [area]?'")
                    continue
                
                elif not user_input:
                    print("💭 Please enter a question or command.")
                    continue
                
                # Get bot response
                print("🤖 Bot: ", end="", flush=True)
                response = bot.chat(user_input, conversation)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("💡 Try rephrasing your question or type 'help' for examples.")
                
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("🔧 Make sure:")
        print("   1. You have a './Datasets' folder")
        print("   2. The folder contains JSON data files")
        print("   3. You have a .env file with MISTRAL_API_KEY")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        print("🔧 Check your configuration and try again.")

if __name__ == "__main__":
    main()