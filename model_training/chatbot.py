from llama_cpp import Llama
import redis

# Load the fine-tuned .gguf model
model_path = "./qwen2.5-3b-research-qa-lora.gguf"  # Replace with the correct path
llm = Llama(model_path=model_path, n_ctx=2048)  # Increase context size if needed

# Initialize Redis for chat history
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# Function to generate a response
def generate_response(prompt, chat_history):
    # Limit chat history length (avoid excessive token usage)
    max_history_length = 10  # Keep last 10 exchanges
    chat_history = chat_history[-max_history_length:]

    # Combine chat history with the new prompt
    full_prompt = "\n".join(chat_history + [f"User: {prompt}"]) + "\nChatbot:"

    # Generate response
    output = llm.create_completion(
        prompt=full_prompt,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        stop=["User:", "\nUser:"],  # Ensure bot stops before next user input
        echo=False
    )

    return output["choices"][0]["text"].strip()

# Chat loop
def chat():
    chat_id = "user_123"  # Unique ID for the chat session
    chat_history = redis_client.lrange(chat_id, 0, -1)  # Load chat history
    chat_history = [msg.decode("utf-8") for msg in chat_history]

    print("Chatbot: Hello! How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        # Generate response
        response = generate_response(user_input, chat_history)
        print(f"Chatbot: {response}")

        # Update chat history (keeping only recent interactions)
        chat_history.append(f"User: {user_input}")
        chat_history.append(f"Chatbot: {response}")
        redis_client.rpush(chat_id, f"User: {user_input}", f"Chatbot: {response}")

        # Trim Redis history to avoid excessive memory usage
        redis_client.ltrim(chat_id, -20, -1)  # Keep last 20 messages

# Start the chat
if __name__ == "__main__":
    chat()
