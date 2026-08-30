"""
capstone_streamlit.py — Bharat Travel Concierge
Run: streamlit run capstone_streamlit.py
"""
import streamlit as st
import uuid
import os
import re
import chromadb
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ── 1. Configuration & State Definition ──────────────────
DOMAIN_NAME = "Bharat Yatra Sahayak"
DOMAIN_DESCRIPTION = "Your AI guide to the 28 states of India with real-time budget calculation."

class CapstoneState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int

# ── 2. Load Models & Knowledge Base (Cached) ──────────────
@st.cache_resource
def load_agent_and_data():
    # Use 8b model to avoid the 429 RateLimit errors you saw earlier
    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=st.secrets["GROQ_API_KEY"],temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize ChromaDB
    client = chromadb.Client()
    try: client.delete_collection("capstone_kb")
    except: pass
    collection = client.create_collection("capstone_kb")

    DOCUMENTS = [
        {"id": "doc_001", "topic": "Destinations: North & Central", "text": "States: Delhi, Punjab, Haryana, Himachal Pradesh, Uttarakhand, Uttar Pradesh, Rajasthan, Madhya Pradesh, Chhattisgarh. Top Places: Taj Mahal (UP), Jaipur Palaces (Rajasthan), Shimla (HP), Varanasi (UP)."},
        {"id": "doc_002", "topic": "Destinations: West & South", "text": "States: Gujarat, Maharashtra, Goa, Karnataka, Kerala, Tamil Nadu, Andhra Pradesh, Telangana, Jharkhand. Top Places: Munnar (Kerala), Hampi (Karnataka), Goa Beaches, Ajanta Caves (Maharashtra)."},
        {"id": "doc_003", "topic": "Destinations: Odisha & Northeast", "text": "States: Odisha, West Bengal, Bihar, Sikkim, Assam, Arunachal, Nagaland, Manipur, Mizoram, Tripura. Top Places: Jagannath Temple (Odisha), Konark Sun Temple, Tea Gardens (Assam), Tawang (Arunachal)."},
        {"id": "doc_004", "topic": "Budgeting: North & Central", "text": "Daily spend for Rajasthan and HP is ₹6,000. UP and MP are approx ₹4,500 per person per day."},
        {"id": "doc_005", "topic": "Budgeting: West & South", "text": "Daily spend for Kerala, Karnataka, and Tamil Nadu is ₹5,000. Mumbai (Maharashtra) is higher at ₹8,000 per day."},
        {"id": "doc_006", "topic": "Budgeting: Odisha & Northeast", "text": "Odisha is economical at ₹4,000. Northeast states (Sikkim/Arunachal) are ₹7,000 due to SUV rentals and permits."},
        {"id": "doc_007", "topic": "Accommodation: North & Central", "text": "Rajasthan: Heritage Havelis. Uttarakhand: Yoga Ashrams. MP: Jungle Resorts."},
        {"id": "doc_008", "topic": "Accommodation: West & South", "text": "Kerala: Houseboats and Ayurvedic Resorts. Goa: Beach Shacks and Villas."},
        {"id": "doc_009", "topic": "Accommodation: Odisha & Northeast", "text": "Odisha: Pilgrim guest houses. Northeast: Eco-homestays and tea estate bungalows."},
        {"id": "doc_010", "topic": "Best Timing", "text": "Nov-Feb: Best for plains/coasts (Goa, Kerala, Rajasthan). Apr-Jun: Best for Himalayas (Sikkim, Himachal)."}
    ]

    texts = [d["text"] for d in DOCUMENTS]
    collection.add(
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),
        ids=[d["id"] for d in DOCUMENTS],
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
    )

    # ── 3. Define Graph Nodes ─────────────────────────────
    
    def memory_node(state: CapstoneState):
        return {"messages": state.get("messages", [])}

    def router_node(state: CapstoneState):
        q = state["question"].lower()
        # Check for time-related questions first
        if any(word in q for word in ["time", "current time", "what time", "what's the time", "tell me the time", "date", "today", "day of week", "clock"]):
            return {"route": "time"}
        if any(word in q for word in ["budget", "cost", "total", "calculate", "people"]):
            return {"route": "tool"}
        if any(word in q for word in ["visit", "where", "place", "state", "recommend", "stay", "accommodation"]):
            return {"route": "retrieve"}
        return {"route": "skip"}

    def retrieval_node(state: CapstoneState):
        q_emb = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        context = "\n".join([f"[{m['topic']}] {d}" for d, m in zip(results["documents"][0], results["metadatas"][0])])
        return {"retrieved": context, "sources": [m["topic"] for m in results["metadatas"][0]]}

    def time_node(state: CapstoneState):
        """Handle real-time information requests"""
        now = datetime.now()
        
        # Format time in formal, precise manner
        formatted_time = now.strftime("%H:%M:%S")  # HH:MM:SS format
        formatted_date = now.strftime("%A, %B %d, %Y")  # Full date
        day_of_week = now.strftime("%A")
        
        time_info = (
            f"Current Date and Time Information:\n"
            f"• Date: {formatted_date}\n"
            f"• Day: {day_of_week}\n"
            f"• Time: {formatted_time} IST (Indian Standard Time)\n"
            f"• Timezone: UTC+5:30"
        )
        
        return {"tool_result": time_info}

    def tool_node(state: CapstoneState):
        q = state["question"]
        # Default fallback values
        people = 1; days = 1; rate = 5000
        
        # Regex extraction
        p_match = re.search(r'(\d+)\s*people', q)
        d_match = re.search(r'(\d+)\s*day', q)
        
        if p_match: people = int(p_match.group(1))
        if d_match: days = int(d_match.group(1))
        
        # Determine rate based on state mention
        if "kerala" in q.lower() or "karnataka" in q.lower(): rate = 5000
        elif "northeast" in q.lower() or "sikkim" in q.lower(): rate = 7000
        elif "odisha" in q.lower(): rate = 4000
        
        subtotal = people * days * rate
        buffer = subtotal * 0.15
        grand_total = subtotal + buffer
        
        report = f"Total for {people} people for {days} days: ₹{grand_total} (Rate: ₹{rate}/day + 15% buffer)"
        return {"tool_result": report}

    def answer_node(state: CapstoneState):
        context = state.get("retrieved", "")
        tool = state.get("tool_result", "")
        route = state.get("route", "")
        
        # Special handling for time-related queries
        if route == "time":
            system_prompt = "You are the Bharat Yatra Sahayak. For time-related queries, provide formal and precise responses based on the provided time information."
            user_msg = f"Time Information Available:\n{tool}\n\nUser Question: {state['question']}\n\nProvide a formal and precise response about the current date and time."
        else:
            system_prompt = "You are the Bharat Yatra Sahayak. Answer strictly based on the provided context or tool results. If location info is missing, say you don't know. Be formal and precise in your responses."
            user_msg = f"Context: {context}\nTool Result: {tool}\nQuestion: {state['question']}"
        
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg)])
        return {"answer": response.content}

    def eval_node(state: CapstoneState):
        answer = state.get("answer", "")
        context = state.get("retrieved", "")[:3000] # Expanded context
        # Skip evaluation for tool and time queries (they are factual/computational)
        if not context or state.get("route") in ["tool", "time"]: return {"faithfulness": 1.0}
        
        prompt = f"Rate faithfulness (0.0 to 1.0). Does the answer use ONLY the context?\nContext: {context}\nAnswer: {answer}"
        result = llm.invoke([HumanMessage(content=prompt)]).content
        try: score = float(re.findall(r"[\d\.]+", result)[0])
        except: score = 0.5
        return {"faithfulness": score, "eval_retries": state.get("eval_retries", 0) + 1}

    def save_node(state: CapstoneState):
        msgs = state.get("messages", [])
        return {"messages": msgs + [{"role": "assistant", "content": state["answer"]}]}

    # ── 4. Build Graph ────────────────────────────────────
    workflow = StateGraph(CapstoneState)
    workflow.add_node("memory", memory_node)
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("time", time_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("eval", eval_node)
    workflow.add_node("save", save_node)

    workflow.set_entry_point("memory")
    workflow.add_edge("memory", "router")

    workflow.add_conditional_edges("router", lambda x: x["route"], {
        "time": "time",
        "retrieve": "retrieve",
        "tool": "tool",
        "skip": "answer"
    })
    workflow.add_edge("time", "answer")
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("tool", "answer")
    workflow.add_edge("answer", "eval")
    
    workflow.add_conditional_edges("eval", 
        lambda x: "retry" if x["faithfulness"] < 0.7 and x["eval_retries"] < 2 else "save",
        {"retry": "retrieve", "save": "save"}
    )
    workflow.add_edge("save", END)

    app = workflow.compile(checkpointer=MemorySaver())
    return app, collection

# ── 5. Streamlit UI ───────────────────────────────────────
st.set_page_config(page_title=DOMAIN_NAME, page_icon="🤖", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #FF6B35;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.1em;
        margin-bottom: 1.5em;
    }
    .response-box {
    background-color: #ffffff;   /* white background */
    color: #000000 !important;   /* black text */
    border-left: 4px solid #FF6B35;
    padding: 1em;
    border-radius: 5px;
    margin-top: 0.5em;
    }
    .sources-box {
        background-color: #0d1b2a;
        border: 1px solid #004e89;
        padding: 0.8em;
        border-radius: 4px;
        margin-top: 0.5em;
        font-size: 0.9em;
    }
    .sidebar-info {
        background-color: #1e3a5f;
        padding: 1em;
        border-radius: 8px;
        margin: 1em 0;
        border-left: 4px solid #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🤖 Bharat Yatra Sahayak</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your AI guide to 28 Indian states with real-time budget calculation</div>', unsafe_allow_html=True)

try:
    agent_app, collection = load_agent_and_data()
    st.sidebar.success(f"✅ KB Ready: {collection.count()} Documents")
except Exception as e:
    st.error(f"Failed to load agent: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

with st.sidebar:
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.subheader("🌏 Coverage")
    st.write("**All 28 Indian States** including:")
    states_by_region = {
        "North": "Delhi, Punjab, Haryana, Himachal Pradesh, Uttarakhand, Uttar Pradesh",
        "Northeast": "Assam, Arunachal Pradesh, Nagaland, Manipur, Mizoram, Tripura, Sikkim, Meghalaya",
        "West": "Gujarat, Maharashtra, Rajasthan, Goa",
        "South": "Karnataka, Kerala, Tamil Nadu, Telangana, Andhra Pradesh",
        "Central": "Madhya Pradesh, Chhattisgarh",
        "East": "West Bengal, Bihar, Jharkhand, Odisha"
    }
    for region, states in states_by_region.items():
        st.write(f"**{region}:** {states}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.subheader("💡 Example Questions")
    example_prompts = [
        "What are the best places to visit in Kerala?",
        "Budget for 2 people for 5 days in Rajasthan",
        "Best time to visit Himachal Pradesh",
        "Accommodation options in Goa"
    ]
    for i, prompt in enumerate(example_prompts, 1):
        st.caption(f"{i}. {prompt}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown('<div class="sidebar-info">', unsafe_allow_html=True)
    st.subheader("📋 Session Info")
    st.write(f"**Thread ID:** `{st.session_state.thread_id}`")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        st.write(msg["content"])
        # Display sources if available
        if "sources" in msg and msg["sources"]:
            st.markdown(f'<div class="sources-box">📚 Sources: {", ".join(msg["sources"])}</div>', unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me about destinations, budgets, best times to visit, or accommodations..."):
    st.chat_message("user", avatar="👤").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Analyzing your question... Consulting travel records..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            try:
                # Add small delay to prevent rapid-fire TPM limit hits
                time.sleep(1)
                result = agent_app.invoke({"question": prompt, "messages": st.session_state.messages[:-1]}, config=config)
                answer = result.get("answer", "I couldn't find a specific answer for that. Please try rephrasing your question.")
                
                st.markdown(f'<div class="response-box">{answer}</div>', unsafe_allow_html=True)
                
                faith = result.get("faithfulness", 0.0)
                sources = result.get("sources", [])
                
                # Display quality metrics and sources
                if sources:
                    sources_str = ", ".join([s.replace("Destinations: ", "").replace("Budgeting: ", "").replace("Accommodation: ", "").replace("Best Timing", "Timing") for s in sources])
                    st.markdown(f'<div class="sources-box">🎯 Confidence: {faith:.1%} | 📚 Sources: {sources_str}</div>', unsafe_allow_html=True)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources,
                    "faithfulness": faith
                })
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    st.warning("⏱️ Rate limit reached. Please wait a moment before asking again.")
                elif "api_key" in error_msg.lower() or "groq" in error_msg.lower():
                    st.error("❌ API Configuration Error. Please check your GROQ_API_KEY in .env file.")
                else:
                    st.error(f"❌ Error: {error_msg}")
                    st.info("💡 Tip: Try rephrasing your question about Indian travel destinations or budgets.")
