from agents import run_feedback_agent
import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq

from sentiment_tagging import FeedbackTagger, TOPICS
from ingest_feedback import build_documents
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


st.set_page_config(
    page_title="Customer Feedback RAG",
    layout="wide"
)

st.title("📊 Customer Feedback Analysis & Q&A")


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )


@st.cache_resource
def load_tagger():
    return FeedbackTagger()


@st.cache_resource
def get_groq_client():
    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )


if "tagged_df" not in st.session_state:
    st.session_state.tagged_df = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


uploaded_file = st.file_uploader(
    "Upload customer feedback (CSV with a 'text' column)",
    type="csv"
)


if uploaded_file and st.button("Process feedback"):

    with st.spinner("Reading feedback data..."):
        df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error(
            "The uploaded CSV must contain a column named 'text'."
        )
        st.stop()

    if df.empty:
        st.error("The uploaded CSV is empty.")
        st.stop()

    with st.spinner(
        "Tagging sentiment and topic for each row..."
    ):
        tagger = load_tagger()
        tagged_df = tagger.tag_dataframe(df)
        st.session_state.tagged_df = tagged_df

    with st.spinner("Building searchable index..."):

        embeddings = load_embeddings()

        docs = build_documents(tagged_df)

        st.session_state.vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

    st.success(
        f"Processed {len(tagged_df)} feedback entries."
    )


if st.session_state.tagged_df is not None:

    df = st.session_state.tagged_df

    col1, col2 = st.columns(2)

    with col1:

        sentiment_counts = (
            df["sentiment"]
            .value_counts()
            .reset_index()
        )

        sentiment_counts.columns = [
            "sentiment",
            "count"
        ]

        fig1 = px.pie(
            sentiment_counts,
            names="sentiment",
            values="count",
            title="Overall Sentiment"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with col2:

        topic_sentiment = (
            df.groupby(
                ["topic", "sentiment"]
            )
            .size()
            .reset_index(name="count")
        )

        fig2 = px.bar(
            topic_sentiment,
            x="topic",
            y="count",
            color="sentiment",
            barmode="group",
            title="Sentiment by Topic"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.divider()

    st.subheader(
        "Ask a question about the feedback"
    )

    fc1, fc2 = st.columns(2)

    with fc1:

        sentiment_filter = st.selectbox(
            "Filter by sentiment",
            [
                "Any",
                "POSITIVE",
                "NEGATIVE"
            ]
        )

    with fc2:

        topic_filter = st.selectbox(
            "Filter by topic",
            ["Any"] + TOPICS
        )

    question = st.text_input(
        "Your question",
        placeholder=(
            "What are customers unhappy about with shipping?"
        )
    )

    if question and st.button("Ask"):

    if st.session_state.vectorstore is None:
        st.error(
            "Please process the feedback before asking a question."
        )
        st.stop()

    sentiment_override = (
        None
        if sentiment_filter == "Any"
        else sentiment_filter
    )

    topic_override = (
        None
        if topic_filter == "Any"
        else topic_filter
    )

    with st.spinner("AI agents are analyzing the feedback..."):

        client = get_groq_client()

        result = run_feedback_agent(
            question=question,
            df=st.session_state.tagged_df,
            vectorstore=st.session_state.vectorstore,
            client=client,
            sentiment_override=sentiment_override,
            topic_override=topic_override
        )

    st.subheader("AI Analysis")

    st.write(result["answer"])
    
    with st.expander("🤖 Agent Execution Trace"):

    plan = result["plan"]
    route_labels = {
    "retrieval": "🔍 Retrieval Agent",
    "analytics": "📊 Analytics Agent",
    "both": "🔍 Retrieval Agent + 📊 Analytics Agent"
}

    st.write("**Route selected:**", plan["route"])

    st.write(
        "**Detected sentiment:**",
        plan["sentiment"] or "Any"
    )

    st.write(
        "**Detected topic:**",
        plan["topic"] or "Any"
    )

    st.write(
        "**Recommendations requested:**",
        "Yes" if plan["needs_recommendation"] else "No"
    )

        with st.expander("Source feedback used"):

            for doc in docs:

                st.write(
                    f"**{doc.metadata['sentiment']} / "
                    f"{doc.metadata['topic']}** — "
                    f"{doc.page_content}"
                )
