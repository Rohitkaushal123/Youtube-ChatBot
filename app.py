import os
import streamlit as st
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load GROQ key from Streamlit secrets (for cloud deployment)
groq_api = st.secrets["GROQ_API_KEY"]

# Extract video ID from full URL or raw ID
def extract_video_id(url_or_id):
    if len(url_or_id) == 11 and "http" not in url_or_id:
        return url_or_id
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url_or_id)
    if match:
        return match.group(1)
    return None

# Streamlit UI
st.title("🎬 YouTube AI Chatbot with Groq")
st.markdown("Ask questions based on the transcript of any YouTube video with captions.")

video_input = st.text_input("🔗 Enter YouTube Video URL or ID:")
user_question = st.text_input("❓ Your question about the video:")

if st.button("Get Answer"):
    if not video_input or not user_question:
        st.warning("Please enter both the video link and a question.")
        st.stop()

    video_id = extract_video_id(video_input)
    if not video_id:
        st.error("❌ Could not extract video ID. Please check your input.")
        st.stop()

    with st.spinner("Processing video transcript..."):
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            transcript = "".join(chunk["text"] for chunk in transcript_list)
        except TranscriptsDisabled:
            st.error("❌ Captions are disabled for this video.")
            st.stop()
        except NoTranscriptFound:
            st.error("❌ No transcript found for this video in English.")
            st.stop()

        # LangChain pipeline
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = splitter.create_documents([transcript])
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents, embedding)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # Prompt template
        prompt = PromptTemplate(
            template="""
You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

Transcript Context:
-------------------
{context}

Question: {question}
""",
            input_variables=["context", "question"]
        )

        # Groq LLM
        llm = ChatGroq(
            model_name="llama3-70b-8192",
            groq_api_key=groq_api
        )

        # Retrieve relevant chunks and generate answer
        docs = retriever.invoke(user_question)
        context = "\n\n".join(doc.page_content for doc in docs)
        final_prompt = prompt.format(context=context, question=user_question)
        response = llm.invoke(final_prompt)

        st.success("✅ Answer:")
        st.markdown(response.content)
