from agents import run_feedback_agent

import streamlit as st
import pandas as pd
import plotly.express as px

from groq import Groq

from sentiment_tagging import FeedbackTagger, TOPICS
from ingest_feedback import build_documents

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Feedback Intelligence Platform")

st.caption(
    "AI-powered customer feedback analysis using "
    "RAG, sentiment analysis, topic classification, "
    "analytics, and AI agents."
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en"
    )


# ============================================================
# LOAD FEEDBACK TAGGER
# ============================================================

@st.cache_resource
def load_tagger():

    return FeedbackTagger()


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource
def get_groq_client():

    return Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )


# ============================================================
# SESSION STATE
# ============================================================

if "tagged_df" not in st.session_state:
    st.session_state.tagged_df = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("1. Upload Customer Feedback")

uploaded_file = st.file_uploader(
    "Upload customer feedback CSV",
    type="csv",
    help="The CSV must contain a column named 'text'."
)


# ============================================================
# PROCESS FEEDBACK
# ============================================================

if uploaded_file and st.button(
    "Process Feedback",
    type="primary"
):

    # --------------------------------------------------------
    # READ CSV
    # --------------------------------------------------------

    with st.spinner(
        "Reading feedback data..."
    ):

        df = pd.read_csv(
            uploaded_file
        )

    # --------------------------------------------------------
    # VALIDATE CSV
    # --------------------------------------------------------

    if "text" not in df.columns:

        st.error(
            "The uploaded CSV must contain "
            "a column named 'text'."
        )

        st.stop()

    if df.empty:

        st.error(
            "The uploaded CSV is empty."
        )

        st.stop()

    # --------------------------------------------------------
    # TAG SENTIMENT + TOPIC
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing sentiment and topics..."
    ):

        tagger = load_tagger()

        tagged_df = tagger.tag_dataframe(
            df
        )

        st.session_state.tagged_df = (
            tagged_df
        )

    # --------------------------------------------------------
    # BUILD FAISS VECTOR INDEX
    # --------------------------------------------------------

    with st.spinner(
        "Building semantic search index..."
    ):

        embeddings = load_embeddings()

        docs = build_documents(
            tagged_df
        )

        st.session_state.vectorstore = (
            FAISS.from_documents(
                docs,
                embeddings
            )
        )

    st.success(
        f"Successfully processed "
        f"{len(tagged_df)} feedback entries."
    )


# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.tagged_df is not None:

    df = st.session_state.tagged_df

    st.divider()

    st.subheader(
        "2. Feedback Analytics Dashboard"
    )

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    total_feedback = len(df)

    positive_count = len(
        df[
            df["sentiment"] == "POSITIVE"
        ]
    )

    negative_count = len(
        df[
            df["sentiment"] == "NEGATIVE"
        ]
    )

    unique_topics = (
        df["topic"].nunique()
    )

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )

    metric1.metric(
        "Total Feedback",
        total_feedback
    )

    metric2.metric(
        "Positive",
        positive_count
    )

    metric3.metric(
        "Negative",
        negative_count
    )

    metric4.metric(
        "Topics",
        unique_topics
    )

    st.divider()

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # SENTIMENT PIE CHART
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TOPIC + SENTIMENT BAR CHART
    # --------------------------------------------------------

    with col2:

        topic_sentiment = (
            df.groupby(
                [
                    "topic",
                    "sentiment"
                ]
            )
            .size()
            .reset_index(
                name="count"
            )
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

    # ========================================================
    # TAGGED DATA PREVIEW
    # ========================================================

    with st.expander(
        "📄 View Processed Feedback"
    ):

        st.dataframe(
            df,
            use_container_width=True
        )

    st.divider()

    # ========================================================
    # AI AGENT Q&A
    # ========================================================

    st.subheader(
        "3. Ask the AI Feedback Agents"
    )

    st.write(
        "Ask qualitative, analytical, or "
        "business recommendation questions "
        "about the customer feedback."
    )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # QUESTION INPUT
    # --------------------------------------------------------

    question = st.text_input(
        "Your question",
        placeholder=(
            "Example: Why are customers unhappy "
            "with shipping and what should we improve?"
        )
    )

    # --------------------------------------------------------
    # ASK BUTTON
    # --------------------------------------------------------

    if question and st.button(
        "Ask AI Agents",
        type="primary"
    ):

        # ----------------------------------------------------
        # CHECK VECTORSTORE
        # ----------------------------------------------------

        if (
            st.session_state.vectorstore
            is None
        ):

            st.error(
                "Please process the feedback "
                "before asking a question."
            )

            st.stop()

        # ----------------------------------------------------
        # MANUAL FILTER OVERRIDES
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # RUN AGENT SYSTEM
        # ----------------------------------------------------

        with st.spinner(
            "AI agents are analyzing "
            "the feedback..."
        ):

            client = get_groq_client()

            result = run_feedback_agent(
                question=question,
                df=(
                    st.session_state
                    .tagged_df
                ),
                vectorstore=(
                    st.session_state
                    .vectorstore
                ),
                client=client,
                sentiment_override=(
                    sentiment_override
                ),
                topic_override=(
                    topic_override
                )
            )

        # ====================================================
        # FINAL AI ANSWER
        # ====================================================

        st.subheader(
            "🤖 AI Analysis"
        )

        st.write(
            result["answer"]
        )

        # ====================================================
        # AGENT EXECUTION TRACE
        # ====================================================

        with st.expander(
            "🤖 Agent Execution Trace"
        ):

            plan = result["plan"]

            route_labels = {
                "retrieval":
                    "🔍 Retrieval Agent",

                "analytics":
                    "📊 Analytics Agent",

                "both":
                    (
                        "🔍 Retrieval Agent "
                        "+ 📊 Analytics Agent"
                    )
            }

            st.write(
                "**Agents selected:**",
                route_labels.get(
                    plan["route"],
                    plan["route"]
                )
            )

            st.write(
                "**Detected sentiment:**",
                (
                    plan["sentiment"]
                    or "Any"
                )
            )

            st.write(
                "**Detected topic:**",
                (
                    plan["topic"]
                    or "Any"
                )
            )

            st.write(
                "**Recommendations requested:**",
                (
                    "Yes"
                    if plan[
                        "needs_recommendation"
                    ]
                    else "No"
                )
            )

        # ====================================================
        # ANALYTICS AGENT OUTPUT
        # ====================================================

        if (
            result["analytics"]
            is not None
        ):

            with st.expander(
                "📊 Analytics Agent Results"
            ):

                analytics = (
                    result["analytics"]
                )

                # --------------------------------------------
                # METRICS
                # --------------------------------------------

                ac1, ac2, ac3 = (
                    st.columns(3)
                )

                ac1.metric(
                    "Total Feedback",
                    analytics.get(
                        "total_feedback",
                        0
                    )
                )

                ac2.metric(
                    "Matching Feedback",
                    analytics.get(
                        "matching_feedback",
                        0
                    )
                )

                ac3.metric(
                    "Percentage",
                    (
                        f"{analytics.get(
                            'percentage_of_total',
                            0
                        )}%"
                    )
                )

                # --------------------------------------------
                # RAW ANALYTICS DATA
                # --------------------------------------------

                st.write(
                    "### Detailed Analytics"
                )

                st.json(
                    analytics
                )

        # ====================================================
        # RETRIEVAL AGENT SOURCES
        # ====================================================

        if (
            result["retrieval"]
            is not None
        ):

            with st.expander(
                "🔍 Source Feedback Used"
            ):

                retrieved_docs = (
                    result["retrieval"]
                    .get(
                        "docs",
                        []
                    )
                )

                if not retrieved_docs:

                    st.info(
                        "No relevant feedback "
                        "was retrieved."
                    )

                else:

                    for index, doc in enumerate(
                        retrieved_docs,
                        start=1
                    ):

                        sentiment = (
                            doc.metadata.get(
                                "sentiment",
                                "Unknown"
                            )
                        )

                        topic = (
                            doc.metadata.get(
                                "topic",
                                "Unknown"
                            )
                        )

                        st.markdown(
                            f"**Feedback {index}**"
                        )

                        st.write(
                            f"**Sentiment:** "
                            f"{sentiment}"
                        )

                        st.write(
                            f"**Topic:** "
                            f"{topic}"
                        )

                        st.write(
                            doc.page_content
                        )

                        st.divider()


# ============================================================
# EMPTY STATE
# ============================================================

else:

    st.info(
        "Upload a customer feedback CSV "
        "and click **Process Feedback** "
        "to begin."
    )
