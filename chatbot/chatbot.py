import streamlit as st
import sqlite3
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import os
import gc
import time



os.environ["STREAMLIT_SERVER_ENABLE_WATCHER_DEFAULTS"] = "false"  
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 

# Function to measure response time
def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        st.session_state['last_response_time'] = f"{end - start:.2f} seconds"
        return result
    return wrapper

# Load the dataset
@st.cache_data(ttl=3600)
def load_dataset():
    try:
        with open("data/dataset.json", "r") as f:
            dataset = json.load(f)
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return []

# Function to clear GPU memory
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# Load the model and tokenizer
@st.cache_resource
def load_model(use_gpu=True):
    try:
        clear_gpu_memory()
        
        base_model_name = "Qwen/Qwen2.5-3B"
        lora_path = "models/qwen2.5-3b-research-qa-lora"
        
        # Device configuration
        device_map = "auto" if use_gpu and torch.cuda.is_available() else "cpu"
        if device_map == "cpu":
            st.warning("Using CPU for inference. This might be slow.")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load base model with optimized parameters
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device_map != "cpu" else torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, lora_path)
        
        # Optimize for inference
        model.eval()
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Function to generate response
@measure_time
def generate_response(prompt, tokenizer, model, max_tokens=256):
    try:
        # Determine device
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            device = "cuda"
        else:
            device = "cpu"
        
        # Create inputs
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # Generate response with optimized parameters
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,  # Enable KV caching for faster generation
                repetition_penalty=1.2,  # Discourage repetition
            pad_token_id=tokenizer.eos_token_id
)

        
        # Decode and return response
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except torch.cuda.OutOfMemoryError:
        clear_gpu_memory()
        st.error("GPU out of memory. Try using CPU mode or reducing max tokens.")
        return "Sorry, the model ran out of GPU memory. Please try with reduced response length or CPU mode."
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return f"Sorry, I couldn't generate a response. Error: {str(e)}"

# Initialize SQLite database
def init_db():
    try:
        conn = sqlite3.connect('chathistory.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, user_input TEXT, bot_response TEXT)''')
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error initializing database: {e}")

# Save chat history to SQLite
def save_chat(user_input, bot_response):
    try:
        conn = sqlite3.connect('chathistory.db')
        c = conn.cursor()
        c.execute("INSERT INTO chat_history (user_input, bot_response) VALUES (?, ?)", (user_input, bot_response))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

# Load chat history from SQLite
@st.cache_data(ttl=60)
def load_chat_history():
    try:
        conn = sqlite3.connect('chathistory.db')
        c = conn.cursor()
        c.execute("SELECT user_input, bot_response FROM chat_history")
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        return []

# Clear chat history from SQLite
def clear_chat_history():
    try:
        conn = sqlite3.connect('chathistory.db')
        c = conn.cursor()
        c.execute("DELETE FROM chat_history")
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error clearing chat history: {e}")
        return False

# Handle form submission
def handle_user_input():
    user_question = st.session_state.new_question_input
    if user_question:
        # Store the question so we can process it after the rerun
        st.session_state['pending_question'] = user_question
        # Clear the input by forcing a rerun with a different key
        st.session_state['input_key'] = f"input_{int(time.time())}"

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Research QA Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Research QA Chatbot")
    st.subheader("Powered by Qwen 2.5-3B")

    # Initialize session states
    if 'last_response_time' not in st.session_state:
        st.session_state['last_response_time'] = ""
    
    if 'input_key' not in st.session_state:
        st.session_state['input_key'] = "new_question_input"
    
    if 'pending_question' not in st.session_state:
        st.session_state['pending_question'] = ""

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # GPU/CPU toggle
        use_gpu = st.checkbox("Use GPU (if available)", value=torch.cuda.is_available())
        
        # Response length control
        max_tokens = st.slider("Maximum response length", 
                               min_value=64, 
                               max_value=512, 
                               value=256,
                               help="Lower values generate faster responses")
        
        # Model loading button
        if st.button("Load/Reload Model"):
            with st.spinner("Loading model... This may take a few minutes."):
                tokenizer, model = load_model(use_gpu=use_gpu)
                if tokenizer is not None and model is not None:
                    st.session_state['tokenizer'] = tokenizer
                    st.session_state['model'] = model
                    st.success("Model loaded successfully!")
                else:
                    st.error("Failed to load model.")
        
        # Clear chat history button
        if st.button("Clear Chat History"):
            if clear_chat_history():
                st.success("Chat history cleared successfully!")
                st.session_state['chat_history'] = []
                st.rerun()
        
        # Display last response time
        if st.session_state['last_response_time']:
            st.info(f"Last response time: {st.session_state['last_response_time']}")
    
    # Initialize database
    init_db()
    
    # Load dataset
    dataset = load_dataset()
    
    # Initialize model in session state if not already loaded
    if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
        with st.spinner("Loading model... This may take a few minutes."):
            tokenizer, model = load_model(use_gpu=use_gpu)
            if tokenizer is not None and model is not None:
                st.session_state['tokenizer'] = tokenizer
                st.session_state['model'] = model
                st.success("Model loaded successfully!")
            else:
                st.error("Failed to load model. Please check logs and try again.")
    
    # Get references to model and tokenizer
    tokenizer = st.session_state.get('tokenizer')
    model = st.session_state.get('model')
    
    if tokenizer is None or model is None:
        st.warning("Model not loaded. Please use the 'Load/Reload Model' button in the sidebar.")
        return

    # Load or initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = load_chat_history()
    
    # Split the screen into two columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat history
        st.subheader("Chat History")
        
        # Create a container for chat messages
        chat_container = st.container()
        
        with chat_container:
            for i, (user_input, bot_response) in enumerate(st.session_state['chat_history']):
                # User message
                st.markdown(f"**You:**")
                st.text_area("User Input", value=user_input, height=80, key=f"user_input_{i}", disabled=True, label_visibility="collapsed")
                
                # Bot response
                st.markdown(f"**Bot:**")
                st.text_area("Bot Response", value=bot_response, height=150, key=f"bot_response_{i}", disabled=True, label_visibility="collapsed")
                
                st.divider()

    with col2:
        # User input section
        st.subheader("Ask a Question")
        
        # Use a form to properly handle input clearing
        with st.form(key="question_form", clear_on_submit=True):
            user_input = st.text_input(
                "Question", 
                placeholder="Type your question here...", 
                key=st.session_state['input_key']
            )
            submit_button = st.form_submit_button("Submit")
        
        # Process any pending question
        if st.session_state.get('pending_question'):
            user_input = st.session_state['pending_question']
            st.session_state['pending_question'] = ""  # Clear the pending question
            
            # Process the question
            with st.spinner("Generating response..."):
                # First check if question is in dataset (fast path)
                bot_response = None
                for entry in dataset:
                    if user_input.lower() == entry["question"].lower():
                        bot_response = entry["answer"]
                        st.success("Found answer in dataset!")
                        break

                # If not found in dataset, generate response
                if not bot_response:
                    bot_response = generate_response(user_input, tokenizer, model, max_tokens=max_tokens)

                # Save to database and update session state
                save_chat(user_input, bot_response)
                st.session_state['chat_history'].append((user_input, bot_response))
                
                # Force a rerun to update the UI
                st.rerun()

        # Handle the form submission
        if submit_button and user_input:
            st.session_state['pending_question'] = user_input
            st.rerun()

# Entry point
if __name__ == "__main__":

    main()
