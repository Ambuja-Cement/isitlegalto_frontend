import os
import streamlit as st
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import HuggingFaceEmbeddings
from ingest import load_documents, process_documents
from privateGPT import main as privateGPT_main
import time

st.set_page_config(
    page_title="PrivateGPT",
    page_icon=":guardsman:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/imartinez/privateGPT',
        'Report a bug': "https://github.com/imartinez/privateGPT/issues",
        'About': "# PrivateGPT: Ask questions to your documents without an internet connection",
    },
)

# Set up page style
st.markdown(
    """
    <style>
    body {
        background-color: #fff;
        font-family: 'Courier New', Courier, monospace;
    }
    h1.custom-h1 {
        color: #ffa500;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    h1.custom-h1 span.legal {
        color: #000;
    }
    h1.custom-h1 span.rest {
        color: #ffa500;
        font-family: 'Courier New', Courier, monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h1 class="custom-h1">
        <span class="legal">isit</span><span class="rest">legalto</span>
    </h1>
    """,
    unsafe_allow_html=True,
)

# User query
query = st.text_input("Enter your question:", key="query")

# Set up environment variables
os.environ["PERSIST_DIRECTORY"] = "db"
os.environ["EMBEDDINGS_MODEL_NAME"] = "all-MiniLM-L6-v2"
os.environ["MODEL"] = "mistral"
os.environ["TARGET_SOURCE_CHUNKS"] = "4"

# Set up language model and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})
llm = Ollama(model="mistral")
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)

# Process uploaded PDF files with PrivateGPT


def privateGPT_main_with_file_path(file_path):
    privateGPT_main(file_path)


# Display answer
if query:
    # Show progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)  # Simulate work
        progress_bar.progress(i + 1)

    res = qa(query)
    answer = st.text_area(
        "Answer:", value=res["result"], height=200, key="answer"
    )

    # Hide progress bar
    progress_bar.empty()

    # Display relevant sources
    if res["source_documents"]:
        for i, doc in enumerate(res["source_documents"]):
            with st.expander(f"{doc.metadata['source']}", expanded=False):
                st.markdown(
                    f"<p style='white-space: pre-wrap;'>{doc.page_content}</p>",
                    unsafe_allow_html=True,
                )
                # Add download button for the source document
                st.download_button(
                    label="Download Source Document",
                    data=doc.metadata["source"],
                    file_name=os.path.basename(doc.metadata["source"]),
                    mime="application/pdf",
                    key=i,  # Add unique key here
                )
                # Add thumbs up and thumbs down buttons
                thumbs_up_button = st.button("Thumbs Up", key=f"thumbs_up_{i}")
                thumbs_down_button = st.button(
                    "Thumbs Down", key=f"thumbs_down_{i}")
                # Handle button clicks
                if thumbs_up_button:
                    st.write("Thanks for the positive feedback!")
                if thumbs_down_button:
                    st.write(
                        "Sorry to hear that. Can you tell us why this answer was not helpful?")
                if i != len(res["source_documents"]) - 1:
                    st.write("---")
