# PDF Chatbot (RAG)

Ask questions about a PDF and get answers based only on its content.

## How it works
1. Load the PDF and split it into 500-character chunks
2. Convert chunks to embeddings (multilingual MiniLM)
3. Store them in ChromaDB
4. Find the 3 chunks closest to the question
5. Send them to an LLM (Groq, free tier) to write the answer

## Tech
Python, LangChain, ChromaDB, Hugging Face, Groq

## Setup
pip install -r requirements.txt
export GROQ_API_KEY="your_key"
python3 app.py

## Example
Question: What should the perplexity be in t-SNE?
Answer: (add a screenshot of your output here)

If the answer is not in the PDF, the bot says so instead of guessing.