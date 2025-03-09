# Research QA Chatbot

A specialized question-answering system built on the Qwen 2.5 3B model fine-tuned for technical AI research topics.

## Overview

This application provides an AI-powered research assistant capable of accurately answering technical questions based on AI research literature. It features:

- Fine-tuned Qwen 2.5 3B model using LoRA (Low-Rank Adaptation)
- Web interface built with Streamlit
- Alternative lightweight CLI using llama.cpp with 4-bit quantization
- Response caching for improved performance
- Comprehensive conversation history

## Installation

### Requirements

- Python 3.9+
- CUDA-compatible GPU recommended (8GB+ VRAM)
- CPU-only mode supported with reduced performance

### Setup

1. Clone the repository
   ```
   git clone https://github.com/sandhavi/Intellihack_CodeLabs_03.git
   cd Intellihack_CodeLabs_03
   cd chatbot       
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Download the model files
   - Place the fine-tuned model in the `models/qwen2.5-3b-research-qa-lora` directory
   - For CLI usage, place the quantized `.gguf` file in the models directory

## Usage

### Streamlit Web Interface

1. Launch the web application:
   ```
   streamlit run chatbot.py
   ```

2. Access the interface in your browser at `http://localhost:8501`

3. Use the sidebar to:
   - Toggle GPU/CPU usage
   - Adjust maximum response length
   - Clear chat history
   - View response time metrics


## Technical Details

- **Base Model**: Qwen 2.5 3B
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Progress**: Initial loss of 7.866800 reduced to 1.546300 over 156 steps
- **Storage**: SQLite for web interface, Redis for CLI
- **Quantization**: 4-bit quantization using llama.cpp for efficient deployment

## Model Performance

The model has been optimized for:
- Accuracy of technical information
- Response speed (average ~1.2s on GPU)
- Memory efficiency (runs with ~4GB RAM)

## Contribution

- Team: CodeLabs
- Task Number: Task-03