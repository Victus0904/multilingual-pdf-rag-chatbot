# Multilingual PDF Chatbot (RAG)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C)
![Llama 2](https://img.shields.io/badge/LLM-Llama%202%207B-blueviolet)
![FAISS](https://img.shields.io/badge/vector%20store-FAISS-0467DF)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Upload PDFs and ask questions about them **in any language**. A Streamlit app built on a
Retrieval-Augmented Generation pipeline: answers come only from the documents, with the source pages listed.

## How it works

```
PDFs ──► PyMuPDF text extraction ──► chunking (300 / overlap 100)
     ──► all-mpnet-base-v2 embeddings ──► FAISS index

Question ──► language detection (langdetect)
         ──► translate to English (M2M100-418M, if needed)
         ──► MMR retrieval (k=3 of 10) ──► Llama-2-7B-Chat answer
         ──► translate back to the user's language ──► answer + source pages
```

- **Multi-PDF:** index several documents at once, with page-level metadata
- **Multilingual:** questions and answers in the user's language via M2M100 translation
- **Diverse context:** Maximal Marginal Relevance retrieval reduces redundant chunks
- **Grounded answers:** a strict prompt, plus a similarity threshold that returns *"Sorry, I couldn't find anything"* instead of guessing
- **Logging:** every Q&A is written to `chat_logs.txt`

## Tech stack

LangChain · FAISS · Hugging Face Transformers · sentence-transformers · Llama 2 · M2M100 · Streamlit · PyMuPDF

## Running it

Dependencies are listed in `requirements.txt` (the notebook also installs them itself).


The notebook `Submission.ipynb` is designed for Google Colab (GPU recommended):

1. Open the notebook and run the install cells.
2. Log in to Hugging Face. Llama 2 is a gated model, so you need approved access.
3. Run the `%%writefile app.py` cell, then the final cell to launch Streamlit through localtunnel.

> The first run downloads Llama-2-7B (~13 GB); loading can take a few minutes.

## License

Code: [MIT](LICENSE). Llama 2 is subject to Meta's
[Llama 2 Community License](https://ai.meta.com/llama/license/).
