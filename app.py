import os
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load .env
load_dotenv()
groq_api = os.environ['GROQ_API_KEY']

# Streamlit UI
st.title("🎬 YouTube Transcript Q&A (Groq-Powered)")

video_id = st.text_input("Enter YouTube Video ID (e.g. Gfr50f6ZBvo):")
question = st.text_input("Ask a question about the video:")

# 👉 Add a button
if st.button("Get Answer"):
    if not video_id or not question:
        st.warning("Please provide both a YouTube Video ID and your question.")
        st.stop()

    with st.spinner("🔍 Processing..."):
        # Fetch transcript
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            transcript = "".join(chunk["text"] for chunk in transcript_list)
        except TranscriptsDisabled:
            st.error("❌ No captions available for this video.")
            st.stop()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = splitter.create_documents([transcript])

        # Embedding & FAISS
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents, embedding)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Groq LLM
        llm = ChatGroq(model_name="llama3-70b-8192", groq_api_key=groq_api)

        # Prompt template
        prompt = PromptTemplate(
            template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}

Question: {question}
""",
            input_variables=['context', 'question']
        )

        # Retrieve relevant context
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Format prompt and get answer
        final_prompt = prompt.format(context=context, question=question)
        response = llm.invoke(final_prompt)

        # Display response
        st.success("✅ Answer:")
        st.markdown(response.content)
