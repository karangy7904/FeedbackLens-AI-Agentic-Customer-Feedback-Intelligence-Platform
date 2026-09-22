import json

MODEL = "openai/gpt-oss-20b"

TOPICS = [
    "shipping and delivery",
    "product quality",
    "customer service",
    "pricing and billing",
    "app or website usability",
]


# ---------------------------------------------------------
# 1. ORCHESTRATOR / PLANNER AGENT
# ---------------------------------------------------------

def plan_question(question, client):
    """
    Decide which tools should handle the user's question.
    """

    system_prompt = f"""
You are an orchestrator for a customer-feedback intelligence system.

Available tools:

1. retrieval
   Use when the user wants qualitative information:
   - customer complaints
   - reasons customers are unhappy
   - examples of feedback
   - recurring problems
   - what customers are saying

2. analytics
   Use when the user asks quantitative questions:
   - how many
   - percentage
   - distribution
   - most common topic
   - sentiment counts

3. both
   Use when answering requires both numerical analysis
   and actual customer feedback.

Available sentiments:
POSITIVE
NEGATIVE

Available topics:
{TOPICS}

Set needs_recommendation=true when the user asks:
- what should we do
- how can we improve
- recommended actions
- possible solutions
"""

    schema = {
        "type": "object",
        "properties": {
            "route": {
                "type": "string",
                "enum": ["retrieval", "analytics", "both"]
            },
            "sentiment": {
                "type": ["string", "null"],
                "enum": ["POSITIVE", "NEGATIVE", None]
            },
            "topic": {
                "type": ["string", "null"],
                "enum": TOPICS + [None]
            },
            "needs_recommendation": {
                "type": "boolean"
            }
        },
        "required": [
            "route",
            "sentiment",
            "topic",
            "needs_recommendation"
        ],
        "additionalProperties": False
    }

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "feedback_plan",
                "strict": True,
                "schema": schema
            }
        },
        temperature=0
    )

    return json.loads(
        response.choices[0].message.content
    )


# ---------------------------------------------------------
# 2. RETRIEVAL AGENT / TOOL
# ---------------------------------------------------------

def retrieval_agent(
    question,
    vectorstore,
    sentiment=None,
    topic=None,
    k=5
):
    filter_dict = {}

    if sentiment:
        filter_dict["sentiment"] = sentiment

    if topic:
        filter_dict["topic"] = topic

    docs = vectorstore.similarity_search(
        question,
        k=k,
        filter=filter_dict or None
    )

    context = "\n".join(
        f"- [{doc.metadata.get('sentiment')} / "
        f"{doc.metadata.get('topic')}] "
        f"{doc.page_content}"
        for doc in docs
    )

    return {
        "docs": docs,
        "context": context
    }


# ---------------------------------------------------------
# 3. ANALYTICS AGENT / TOOL
# ---------------------------------------------------------

def analytics_agent(
    df,
    sentiment=None,
    topic=None
):
    filtered_df = df.copy()

    if sentiment:
        filtered_df = filtered_df[
            filtered_df["sentiment"] == sentiment
        ]

    if topic:
        filtered_df = filtered_df[
            filtered_df["topic"] == topic
        ]

    total_feedback = len(df)
    matching_feedback = len(filtered_df)

    percentage = (
        round(
            matching_feedback / total_feedback * 100,
            2
        )
        if total_feedback
        else 0
    )

    sentiment_counts = (
        filtered_df["sentiment"]
        .value_counts()
        .to_dict()
    )

    topic_counts = (
        filtered_df["topic"]
        .value_counts()
        .to_dict()
    )

    return {
        "total_feedback": total_feedback,
        "matching_feedback": matching_feedback,
        "percentage_of_total": percentage,
        "sentiment_breakdown": sentiment_counts,
        "topic_breakdown": topic_counts
    }


# ---------------------------------------------------------
# 4. INSIGHT / SYNTHESIS AGENT
# ---------------------------------------------------------

def synthesis_agent(
    question,
    plan,
    client,
    retrieval_result=None,
    analytics_result=None
):
    evidence = "No retrieved feedback."

    if retrieval_result:
        evidence = retrieval_result["context"]

    analytics = "No analytics requested."

    if analytics_result:
        analytics = json.dumps(
            analytics_result,
            indent=2
        )

    prompt = f"""
You are a customer-feedback intelligence analyst.

Answer the user's question using ONLY the supplied evidence
and analytics.

User question:
{question}

Agent plan:
{json.dumps(plan, indent=2)}

Analytics:
{analytics}

Retrieved customer feedback:
{evidence}

Instructions:

- Never invent statistics.
- Use analytics numbers exactly as provided.
- Base qualitative claims on retrieved feedback.
- Separate factual findings from recommendations.
- If recommendations were requested, provide practical
  business actions tied to the observed feedback.
- If evidence is insufficient, say so.
- Keep the answer concise but useful.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=700
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# 5. MAIN ORCHESTRATOR
# ---------------------------------------------------------

def run_feedback_agent(
    question,
    df,
    vectorstore,
    client,
    sentiment_override=None,
    topic_override=None
):

    # Step 1: Agent decides what needs to happen
    plan = plan_question(question, client)

    # Manual Streamlit filters can override agent decision
    if sentiment_override:
        plan["sentiment"] = sentiment_override

    if topic_override:
        plan["topic"] = topic_override

    retrieval_result = None
    analytics_result = None

    # Step 2: Dynamically execute tools

    if plan["route"] in ["retrieval", "both"]:

        retrieval_result = retrieval_agent(
            question=question,
            vectorstore=vectorstore,
            sentiment=plan["sentiment"],
            topic=plan["topic"]
        )

    if plan["route"] in ["analytics", "both"]:

        analytics_result = analytics_agent(
            df=df,
            sentiment=plan["sentiment"],
            topic=plan["topic"]
        )

    # Step 3: Synthesize everything
    answer = synthesis_agent(
        question=question,
        plan=plan,
        client=client,
        retrieval_result=retrieval_result,
        analytics_result=analytics_result
    )

    return {
        "answer": answer,
        "plan": plan,
        "retrieval": retrieval_result,
        "analytics": analytics_result
    }