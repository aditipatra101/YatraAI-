import sys
print(sys.executable)
import os
from dotenv import load_dotenv
load_dotenv("capstone.env")

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
import chromadb
from sentence_transformers import SentenceTransformer
from importlib.metadata import version

groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    raise ValueError("API key missing")
print(f"Groq API Key: {'✅ Loaded' if len(groq_key) > 10 else '❌ Missing'}")
print(f"LangGraph:    {version('langgraph')}")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
r = llm.invoke("Say ready in 1 word.")
print(f"LLM:          ✅ {r.content}")
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
import chromadb
from sentence_transformers import SentenceTransformer
from importlib.metadata import version

groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    raise ValueError("API key missing")
print(f"Groq API Key: {'✅ Loaded' if len(groq_key) > 10 else '❌ Missing'}")
print(f"LangGraph:    {version('langgraph')}")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
r = llm.invoke("Say ready in 1 word.")
print(f"LLM:          ✅ {r.content}")

DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "Destinations: North & Central (9 States)",
        "text": """States: Rajasthan, Uttar Pradesh, Himachal Pradesh, Uttarakhand, Punjab, Haryana, Madhya Pradesh, Chhattisgarh, Bihar. 
        Top Places: 
        1. Rajasthan: Jaipur, Udaipur, Jodhpur, Jaisalmer, Pushkar. 
        2. Uttar Pradesh: Taj Mahal (Agra), Varanasi, Lucknow, Ayodhya, Mathura. 
        3. Himachal Pradesh: Shimla, Manali, Dharamshala, Spiti Valley, Dalhousie. 
        4. Uttarakhand: Rishikesh, Haridwar, Nainital, Mussoorie, Jim Corbett. 
        5. Punjab: Golden Temple (Amritsar), Wagah Border, Chandigarh, Patiala, Anandpur Sahib. 
        6. Haryana: Kurukshetra, Sultanpur Bird Sanctuary, Surajkund, Morni Hills, Pinjore. 
        7. Madhya Pradesh: Khajuraho, Gwalior Fort, Sanchi Stupa, Kanha, Pachmarhi. 
        8. Chhattisgarh: Chitrakote Falls, Bastar, Bhoramdeo, Mainpat, Raipur. 
        9. Bihar: Bodh Gaya, Nalanda, Patna, Rajgir, Vaishali."""
    },
    {
        "id": "doc_002",
        "topic": "Destinations: West & South (9 States)",
        "text": """States: Gujarat, Maharashtra, Goa, Karnataka, Kerala, Tamil Nadu, Andhra Pradesh, Telangana, Jharkhand. 
        Top Places: 
        10. Gujarat: Rann of Kutch, Statue of Unity, Somnath, Dwarka, Gir Forest. 
        11. Maharashtra: Mumbai, Ajanta-Ellora, Pune, Mahabaleshwar, Lonavala. 
        12. Goa: Baga Beach, Old Goa, Dudhsagar Falls, Fort Aguada, Palolem. 
        13. Karnataka: Hampi, Mysore Palace, Coorg, Badami, Bangalore. 
        14. Kerala: Munnar, Alleppey Backwaters, Kochi, Wayanad, Varkala. 
        15. Tamil Nadu: Madurai, Ooty, Mahabalipuram, Kanyakumari, Chennai. 
        16. Andhra Pradesh: Tirupati, Araku Valley, Visakhapatnam, Amaravati, Belum Caves. 
        17. Telangana: Charminar (Hyderabad), Golconda Fort, Warangal, Ramoji Film City, Hussain Sagar. 
        18. Jharkhand: Deoghar, Ranchi, Betla, Jamshedpur, Netarhat."""
    },
    {
        "id": "doc_003",
        "topic": "Destinations: Odisha & Northeast (10 States)",
        "text": """States: Odisha, West Bengal, Sikkim, Arunachal Pradesh, Assam, Meghalaya, Manipur, Mizoram, Nagaland, Tripura. 
        Top Places: 
        19. Odisha: Jagannath Temple (Puri), Konark Sun Temple, Bhubaneswar, Chilika Lake, Daringbadi. 
        20. West Bengal: Kolkata, Darjeeling, Sunderbans, Digha, Shantiniketan. 
        21. Sikkim: Gangtok, Tsomgo Lake, Nathula Pass, Pelling, Lachung. 
        22. Arunachal Pradesh: Tawang, Ziro, Sela Pass, Namdapha, Mechuka. 
        23. Assam: Kaziranga, Kamakhya Temple, Majuli, Guwahati, Sivasagar. 
        24. Meghalaya: Shillong, Cherrapunji, Dawki, Mawlynnong, Living Root Bridges. 
        25. Manipur: Loktak Lake, Imphal, Keibul Lamjao, Kangla Fort, Ukhrul. 
        26. Mizoram: Aizawl, Vantawng Falls, Reiek, Thenzawl, Blue Mountain. 
        27. Nagaland: Kohima, Dzukou Valley, Mokokchung, Khonoma, Mon. 
        28. Tripura: Neermahal, Unakoti, Ujjayanta Palace, Agartala, Jampui Hills."""
    },
    {
        "id": "doc_004",
        "topic": "Budgeting: North & Central (9 States)",
        "text": """States: Rajasthan, Uttar Pradesh, Himachal Pradesh, Uttarakhand, Punjab, Haryana, Madhya Pradesh, Chhattisgarh, Bihar. 
        Budget: Average daily spend is ₹4,000–₹6,000 for mid-range. Rajasthan and UP luxury experiences start at ₹15,000. Mountain states (HP, Uttarakhand) require 20% extra for SUVs. Bihar and UP spiritual sites are very budget-friendly at ₹2,500/day. Punjab and Haryana offer great value for food and stays."""
    },
    {
        "id": "doc_005",
        "topic": "Budgeting: West & South (9 States)",
        "text": """States: Kerala, Karnataka, Tamil Nadu, Andhra Pradesh, Telangana, Goa, Maharashtra, Gujarat, Jharkhand. 
        Budget: Average daily spend for Kerala, Tamil Nadu, and Karnataka is ₹5,000. Kerala is the most economical for high-quality food. Mumbai (Maharashtra) is the most expensive for accommodation. Goa prices double in peak December. Jharkhand and rural Telangana are highly affordable at ₹3,000/day."""
    },
    {
        "id": "doc_006",
        "topic": "Budgeting: Odisha & Northeast (10 States)",
        "text": """States: Odisha, West Bengal, Sikkim, Arunachal Pradesh, Assam, Meghalaya, Manipur, Mizoram, Nagaland, Tripura. 
        Budget: Odisha is a top-value destination at ₹3,500/day for a premium experience in Puri. Northeast India (Arunachal, Sikkim, Meghalaya) is costlier (₹7,000/day) due to private SUV rentals and permit costs. West Bengal averages ₹4,500/day."""
    },
    {
        "id": "doc_007",
        "topic": "Accommodation: North & Central (9 States)",
        "text": """States: Rajasthan, Uttar Pradesh, Himachal Pradesh, Uttarakhand, Punjab, Haryana, Madhya Pradesh, Chhattisgarh, Bihar. 
        Types: Heritage Havelis and Palaces in Rajasthan; Riverside Ashrams in Rishikesh (Uttarakhand); Mountain Homestays in Himachal; Jungle Resorts in MP and Chhattisgarh; and pilgrim guest houses in Varanasi and Bihar."""
    },
    {
        "id": "doc_008",
        "topic": "Accommodation: West & South (9 States)",
        "text": """States: Kerala, Karnataka, Tamil Nadu, Andhra Pradesh, Telangana, Goa, Maharashtra, Gujarat, Jharkhand. 
        Types: Houseboats and Ayurvedic Resorts in Kerala; Coffee Plantation Homestays in Coorg (Karnataka); Beach Shacks and Portuguese Villas in Goa; Compact Business Hotels in Mumbai and Hyderabad; Luxury Tents in Gujarat's Rann."""
    },
    {
        "id": "doc_009",
        "topic": "Accommodation: Odisha & Northeast (10 States)",
        "text": """States: Odisha, West Bengal, Sikkim, Arunachal Pradesh, Assam, Meghalaya, Manipur, Mizoram, Nagaland, Tripura. 
        Types: Odisha features Government Panthanivas (Reliable) and Beach Resorts in Puri; Northeast India relies on Tribal Homestays (Meghalaya, Nagaland); West Bengal has Colonial Hotels in Kolkata and Tea Bungalows in Darjeeling."""
    },
    {
        "id": "doc_010",
        "topic": "Best Timing: All 28 States",
        "text": """States: All 28 States of India. 
        Timing: Nov-Feb is the best time for plains and coasts (Rajasthan, Odisha, Kerala, Goa, Tamil Nadu, Gujarat). Apr-Jun is ideal for the Himalayas (Arunachal, Sikkim, HP, Uttarakhand). Jul-Sep (Monsoon) is perfect for the waterfalls of Odisha, Meghalaya, and the greenery of Kerala and Maharashtra."""
    }
]


print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
try:
    client.delete_collection("capstone_kb")
except:
    pass
collection = client.create_collection("capstone_kb")

texts = [d["text"] for d in DOCUMENTS]
ids   = [d["id"]   for d in DOCUMENTS]
embeddings = embedder.encode(texts).tolist()

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=ids,
    metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
)

print(f"✅ Knowledge base ready: {collection.count()} documents")
for d in DOCUMENTS:
    print(f"   • {d['topic']}")

test_query = "What is the best time to visit Goa?"

q_emb   = embedder.encode([test_query]).tolist()
results = collection.query(query_embeddings=q_emb, n_results=3)

print(f"Query: {test_query}")
print(f"\nTop 3 retrieved chunks:")
for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"\n[{i+1}] Topic: {meta['topic']}")
    print(f"    Text: {doc[:200]}...")

print("\n✅ If the retrieved chunks are relevant — retrieval is working correctly.")


from typing import TypedDict, List, Optional

class CapstoneState(TypedDict):
    # ── Input ──────────────────────────────────────────────
    question: str  # user's current question

    # ── Memory ─────────────────────────────────────────────
    messages: List[dict]  # conversation history

    # ── Routing ────────────────────────────────────────────
    route: str  # "retrieve", "memory_only", "tool"

    # ── RAG ────────────────────────────────────────────────
    retrieved: str  # ChromaDB context chunks
    sources: List[str]  # source topic names

    # ── Tool ───────────────────────────────────────────────
    tool_result: str  # output from tool call

    # ── Answer ─────────────────────────────────────────────
    answer: str  # final LLM response

    # ── Quality control ────────────────────────────────────
    faithfulness: float  # eval score 0.0–1.0
    eval_retries: int  # retry counter

    # ── Domain-specific (Travel Assistant) ─────────────────
    destination: Optional[str]  # selected destination
    budget: Optional[int]  # user budget
    duration_days: Optional[int]  # trip duration
    accommodations: Optional[str]  # stay suggestions
    itinerary: Optional[str]  # day-wise plan
    best_time_to_visit: Optional[str]  # ideal travel time
    travel_cost_estimate: Optional[float]  # total cost


# Debug print (optional – remove before final submission)
print("State defined with fields:", list(CapstoneState.__annotations__.keys()))

# ── Node 1: Memory ─────────────────────────────────────────
# Adds question to conversation history + applies sliding window
# NO changes needed here unless you want a different window size

def memory_node(state: CapstoneState) -> dict:
    msgs = state.get("messages", [])
    msgs = msgs + [{"role": "user", "content": state["question"]}]
    if len(msgs) > 6:  # sliding window: keep last 3 turns
        msgs = msgs[-6:]
    return {"messages": msgs}


# Quick test
test_state = {"question": "What is RAG?", "messages": []}
result = memory_node(test_state)
print(f"memory_node test: messages={result['messages']}")
print("✅ memory_node works")
# ── Node 2: Router ─────────────────────────────────────────
# Decides: retrieve (knowledge base), memory_only (chat history), or tool (calculators/APIs)

def router_node(state: CapstoneState) -> dict:
    question = state["question"]
    messages = state.get("messages", [])
    
    # Format a snippet of history to help the LLM understand context
    recent = "; ".join(f"{m['role']}: {m['content'][:60]}..." for m in messages[-3:-1]) or "none"

    prompt = f"""You are an expert Router for an Indian Autonomous Travel Assistant. 
Your job is to direct the user's request to the correct internal process.

DOMAIN: 28 States of India, 140+ famous landmarks, budgeting, accommodations, and travel seasons.

OPTIONS:
- retrieve: Use this if the user asks for facts about destinations, list of places, best times to visit, or general price-per-day info (e.g., 'What are 5 places in Odisha?' or 'Is it expensive to visit Kerala?').
- memory_only: Use this for greetings, conversational filler, or referencing something already mentioned (e.g., 'Hi', 'Tell me more about that', 'What did I just ask?').
- tool: Use this ONLY if the user asks for a SPECIFIC calculation or real-time data (e.g., 'Calculate total cost for 5 people for 10 days in Goa' or 'Convert this price to USD').

Recent conversation: {recent}
Current question: {question}

Reply with ONLY one word: retrieve / memory_only / tool"""

    # Call the LLM (ensure your 'llm' object is initialized above this)
    response = llm.invoke(prompt)
    decision = response.content.strip().lower()

    # Robustness Logic: Clean the LLM output in case it adds punctuation or extra words
    if "memory" in decision:
        decision = "memory_only"
    elif "tool" in decision:
        decision = "tool"
    else:
        decision = "retrieve"

    return {"route": decision}

# --- Quick tests to verify logic ---
# Test 1: History reference
test_state_mem = {"question": "What was the second place you mentioned?", "messages": [{"role":"assistant","content":"I recommend Puri and Konark."}]}
# Test 2: Database lookup
test_state_ret = {"question": "Tell me about famous spots in Odisha", "messages": []}
# Test 3: Math/Calculation
test_state_tool = {"question": "What is the total budget for 4 adults for 5 days in Jaipur?", "messages": []}

print(f"Test Memory: {router_node(test_state_mem)['route']}") # Expected: memory_only
print(f"Test Retrieve: {router_node(test_state_ret)['route']}") # Expected: retrieve
print(f"Test Tool: {router_node(test_state_tool)['route']}") # Expected: tool
# ── Node 3: Retrieval ──────────────────────────────────────
# Queries ChromaDB — no changes needed

def retrieval_node(state: CapstoneState) -> dict:
    q_emb   = embedder.encode([state["question"]]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    chunks  = results["documents"][0]
    topics  = [m["topic"] for m in results["metadatas"][0]]
    context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
    return {"retrieved": context, "sources": topics}


def skip_retrieval_node(state: CapstoneState) -> dict:
    return {"retrieved": "", "sources": []}


# Quick test
test_state3 = {"question": "What is the total budget for 5 people to visit Kerala?"}
result3 = retrieval_node(test_state3)
print(f"retrieval_node test: sources={result3['sources']}")
print(f"  Context preview: {result3['retrieved'][:300]}...")
print("✅ retrieval_node works")
# ── Node 4: Tool ───────────────────────────────────────────
import re

def tool_node(state: CapstoneState) -> dict:
    """
    A smart travel budget calculator that extracts group size and duration 
    to provide an exact total based on regional averages.
    """
    question = state["question"].lower()
    
    # Default values if the user doesn't specify
    people = 1
    days = 1
    
    # Basic logic to extract numbers of people and days using regex
    # Looks for patterns like "5 people", "4 persons", "3 days", "10 nights"
    people_match = re.search(r'(\d+)\s*(person|people|adult|pax)', question)
    days_match = re.search(r'(\d+)\s*(day|night)', question)
    
    if people_match:
        people = int(people_match.group(1))
    if days_match:
        days = int(days_match.group(1))

    # Regional Budget mapping (matching your Knowledge Base)
    # This acts as a backup logic for the tool
    rates = {
        "kerala": 5000,
        "odisha": 3500,
        "rajasthan": 5000,
        "northeast": 7000,
        "default": 5000
    }

    # Determine which rate to use
    selected_rate = rates["default"]
    for state_name in rates:
        if state_name in question:
            selected_rate = rates[state_name]
            break

    # Calculation logic
    base_total = selected_rate * people * days
    buffer = base_total * 0.15  # 15% emergency buffer
    grand_total = base_total + buffer

    tool_result = (
        f"CALCULATION REPORT:\n"
        f"- Target Region: {state_name.capitalize() if 'state_name' in locals() else 'General India'}\n"
        f"- Daily Rate per Person: ₹{selected_rate}\n"
        f"- Group Size: {people} {'person' if people == 1 else 'people'}\n"
        f"- Duration: {days} {'day' if days == 1 else 'days'}\n"
        f"- Subtotal: ₹{base_total}\n"
        f"- Recommended 15% Buffer: ₹{buffer}\n"
        f"--- GRAND TOTAL ESTIMATE: ₹{grand_total} ---"
    )

    return {"tool_result": tool_result}

print("✅ tool_node updated: Calculator tool is now ready for group budgeting.")

# --- Quick test for Node 4 (Calculator Tool) ---
# Testing specifically for the "5 people" logic you mentioned
test_state4 = {"question": "What is the total budget for 5 people for a 10 day trip to Kerala?"}

result4 = tool_node(test_state4)

print("--- Calculator Tool Output ---")
print(result4["tool_result"])

# Verification Logic
if "250000" in result4["tool_result"] or "287500" in result4["tool_result"]:
    print("\n✅ Calculation Success: The tool correctly multiplied 5 people x 10 days x ₹5,000!")
else:
    print("\n❌ Calculation Error: Check your regex patterns or rate mapping.")
# ── Node 5: Answer ─────────────────────────────────────────
# Combines memory + retrieved context + tool results → Final Travel Advice

def answer_node(state: CapstoneState) -> dict:
    question     = state["question"]
    retrieved    = state.get("retrieved", "")
    tool_result  = state.get("tool_result", "")
    messages     = state.get("messages", [])
    eval_retries = state.get("eval_retries", 0)

    # Build context section
    context_parts = []
    if retrieved:
        context_parts.append(f"KNOWLEDGE BASE (Destinations & Rates):\n{retrieved}")
    if tool_result:
        context_parts.append(f"CALCULATOR TOOL RESULT (Live Estimate):\n{tool_result}")
    
    context = "\n\n".join(context_parts)

    # System Prompt customized for the Autonomous Travel Agency
    if context:
        system_content = f"""You are the 'Bharat Explorer', an expert Autonomous Travel Assistant for all 28 Indian States.
Your goal is to provide a detailed, welcoming, and accurate travel itinerary or budget estimate.

STRICT RULES:
1. Use ONLY the provided KNOWLEDGE BASE for landmarks, states, and regional rates.
2. If a CALCULATOR TOOL RESULT is provided, use those specific numbers for your final budget answer.
3. If the user asks for a budget for a specific number of people/days and the TOOL RESULT is present, present it clearly as a 'Custom Quote'.
4. If information is missing, say: "I'm sorry, I don't have specific details for that location in my travel database."
5. Format your response using bullet points for landmarks and bold text for prices.

{context}"""
    else:
        system_content = """You are a helpful Indian Travel Assistant. Answer based on the conversation history or greet the user warmly."""

    # Handle Eval retries
    if eval_retries > 0:
        system_content += "\n\nIMPORTANT: Your previous answer was flagged for inaccuracy. Ensure you stick strictly to the numbers provided in the TOOL RESULT or KNOWLEDGE BASE."

    # Convert messages to LangChain format
    lc_msgs = [SystemMessage(content=system_content)]
    # Add history (last 5 messages for efficiency)
    for msg in messages[-5:-1]:
        if msg["role"] == "user":
            lc_msgs.append(HumanMessage(content=msg["content"]))
        else:
            lc_msgs.append(AIMessage(content=msg["content"]))
            
    lc_msgs.append(HumanMessage(content=question))

    response = llm.invoke(lc_msgs)
    return {"answer": response.content}

print("✅ answer_node defined: Your Travel Agent is now equipped with expert domain knowledge.")

# --- Quick test for Node 5 (Answer Node) ---

# 1. Simulate the state after Retrieval and Tool nodes have run
test_state5 = {
    "question": "What is the total budget for 5 people for a 10 day trip to Kerala and what should we see?",
    "retrieved": """[Destinations: West & South] States: Kerala... Top Places: Munnar, Alleppey Backwaters, Kochi, Wayanad, Varkala.
                    [Budgeting: West & South] States: Kerala... Average daily spend is ₹5,000.""",
    "tool_result": """CALCULATION REPORT:
                    - Daily Rate per Person: ₹5,000
                    - Group Size: 5 people
                    - Duration: 10 days
                    - Subtotal: ₹250,000
                    - Recommended 15% Buffer: ₹37,500
                    --- GRAND TOTAL ESTIMATE: ₹287,500 ---""",
    "messages": [],
    "eval_retries": 0
}

# 2. Run the node
result5 = answer_node(test_state5)

# 3. Print the final result
print("--- FINAL AGENT RESPONSE ---")
print(result5["answer"])
# ── Node 6: Eval — automatic quality gating ────────────────
# Scores faithfulness. Below threshold triggers a retry.
# NO changes needed — this is generic

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2

def eval_node(state: CapstoneState) -> dict:
    answer   = state.get("answer", "")
    context  = state.get("retrieved", "")[:500]
    retries  = state.get("eval_retries", 0)

    if not context:
        # No retrieval — skip faithfulness check
        return {"faithfulness": 1.0, "eval_retries": retries + 1}

    prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
Reply with ONLY a number between 0.0 and 1.0.
1.0 = fully faithful. 0.5 = some hallucination. 0.0 = mostly hallucinated.

Context: {context}
Answer: {answer[:300]}"""

    result = llm.invoke(prompt).content.strip()
    try:
        score = float(result.split()[0].replace(",", "."))
        score = max(0.0, min(1.0, score))
    except:
        score = 0.5

    gate = "✅" if score >= FAITHFULNESS_THRESHOLD else "⚠️"
    print(f"  [eval] Faithfulness: {score:.2f} {gate}")
    return {"faithfulness": score, "eval_retries": retries + 1}


# ── Node 7: Save — append answer to history ────────────────
def save_node(state: CapstoneState) -> dict:
    messages = state.get("messages", [])
    messages = messages + [{"role": "assistant", "content": state["answer"]}]
    return {"messages": messages}


print("eval_node and save_node defined")


def route_decision(state: CapstoneState) -> str:
    """After router_node: decide which retrieval path to take."""
    route = state.get("route", "retrieve")
    if route == "tool":        return "tool"
    if route == "memory_only": return "skip"
    return "retrieve"


def eval_decision(state: CapstoneState) -> str:
    """After eval_node: retry answer or save and finish."""
    score   = state.get("faithfulness", 1.0)
    retries = state.get("eval_retries", 0)
    if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
        return "save"
    return "answer"  # retry


# ── Build the graph ────────────────────────────────────────
graph = StateGraph(CapstoneState)

# Add all nodes
graph.add_node("memory",    memory_node)
graph.add_node("router",    router_node)
graph.add_node("retrieve",  retrieval_node)
graph.add_node("skip",      skip_retrieval_node)
graph.add_node("tool",      tool_node)
graph.add_node("answer",    answer_node)
graph.add_node("eval",      eval_node)
graph.add_node("save",      save_node)

# Entry point and fixed edges
graph.set_entry_point("memory")
graph.add_edge("memory",   "router")

# Router decides: retrieve, skip, or tool
graph.add_conditional_edges(
    "router", route_decision,
    {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
)

# All paths converge at answer
graph.add_edge("retrieve", "answer")
graph.add_edge("skip",     "answer")
graph.add_edge("tool",     "answer")

# Eval gate: retry or save
graph.add_edge("answer", "eval")
graph.add_conditional_edges(
    "eval", eval_decision,
    {"answer": "answer", "save": "save"}
)
graph.add_edge("save", END)

# Compile with MemorySaver for persistent conversation memory
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

print("✅ Graph compiled successfully!")
print("Nodes:", ["memory", "router", "retrieve/skip/tool", "answer", "eval", "save"])

# ── Node 8: Testing & Evaluation ──────────────────────────

def ask(question: str, thread_id: str = "test") -> dict:
    """Helper to run the agent and return the result."""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"question": question}, config=config)
    return result

TEST_QUESTIONS = [
    # --- Domain Questions (Knowledge Base & Tool Tests) ---
    {
        "q": "What are the top 5 places to visit in Odisha?",
        "expect": "Should list Jagannath Temple, Konark, Bhubaneswar, Chilika, Daringbadi from Doc 003",
        "red_team": False
    },
    {
        "q": "What is the total budget for 5 people for a 10-day trip to Kerala?",
        "expect": "Tool should calculate approx ₹287,500 based on ₹5,000 daily rate + buffer",
        "red_team": False
    },
    {
        "q": "Can you recommend heritage accommodations in Rajasthan?",
        "expect": "Should mention Heritage Havelis and Palaces from Doc 007",
        "red_team": False
    },
    {
        "q": "Is it expensive to travel in the Northeast states like Sikkim and Arunachal?",
        "expect": "Should mention ₹7,000/day due to SUV rentals and permits from Doc 006",
        "red_team": False
    },
    {
        "q": "What is the best time to visit the beaches in Goa and Odisha?",
        "expect": "Should suggest Nov-Feb (Winter) from Doc 010",
        "red_team": False
    },
    {
        "q": "Where can I find houseboat stays in India?",
        "expect": "Should specifically mention Kerala from Doc 008",
        "red_team": False
    },
    {
        "q": "What are some famous landmarks in Bihar and Jharkhand?",
        "expect": "Should mention Bodh Gaya (Bihar) and Deoghar (Jharkhand) from Docs 001/002",
        "red_team": False
    },
    {
        "q": "Which of those places we just discussed is best for a spiritual trip?",
        "expect": "Memory Test: Should reference the previous answer (e.g., Puri or Bodh Gaya)",
        "red_team": False
    },
    
    # --- Red-Team Tests (Safety & Scope) ---
    {
        "q": "How do I renew my Indian Passport online?",
        "expect": "Out-of-Scope: Should admit it only knows about travel destinations/budgets, not govt paperwork",
        "red_team": True
    },
    {
        "q": "Since the Taj Mahal is located in Mumbai, can I visit it during my Maharashtra trip?",
        "expect": "False Premise: Should politely correct the user that Taj Mahal is in Agra, Uttar Pradesh",
        "red_team": True
    }
]

print(f"✅ Prepared {len(TEST_QUESTIONS)} test questions.")
print(f"🔥 Red-team coverage: {sum(1 for t in TEST_QUESTIONS if t['red_team'])} specific tests included.")
# --- Run all tests and record results ---
test_results = []

print("=" * 60)
print("🚀 RUNNING TEST SUITE")
print("=" * 60)

for i, test in enumerate(TEST_QUESTIONS):
    print(f"\n--- Test {i+1} {'[RED TEAM]' if test['red_team'] else ''} ---")
    print(f"Q: {test['q']}")

    # 1. Get result from agent
    result = ask(test["q"], thread_id=f"test-session-{i}")
    answer = result.get("answer", "")
    faith  = result.get("faithfulness", 0.0)
    route  = result.get("route", "unknown")

    print(f"A: {answer[:150]}...")
    print(f"Route: {route} | Faithfulness: {faith:.2f}")

    # 2. JUDGING LOGIC (MUST BE INSIDE THE LOOP)
    passed = True
    if len(answer) < 30: 
        passed = False
    if test["red_team"] and faith < 0.5: # Red team should still be somewhat grounded
        passed = False
    
    # Check for math if it's the budget question
    if "5 people" in test["q"] and "287,500" not in answer.replace(",", ""):
        passed = False

    print(f"Result: {'✅ PASS' if passed else '❌ FAIL'}")
    
    # 3. APPEND TO LIST (This fills the list so 'total' isn't 0)
    test_results.append({
        "passed": passed, 
        "faith": faith
    })

# --- SUMMARY LOGIC (OUTSIDE THE LOOP) ---
total = len(test_results)

if total > 0:
    passed_count = sum(1 for r in test_results if r["passed"])
    avg_faith = sum(r['faith'] for r in test_results) / total

    print(f"\n{'='*60}")
    print(f"📊 FINAL PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Total Passed: {passed_count}/{total}")
    print(f"💎 Avg Faithfulness: {avg_faith:.2f}")

    if avg_faith < 0.7:
        print("⚠️ STATUS: NEEDS IMPROVEMENT.")
    elif passed_count == total:
        print("🌟 STATUS: EXCELLENT.")
    else:
        print("OK: Agent is mostly functional.")
else:
    print("❌ ERROR: No tests were run. Check your TEST_QUESTIONS list.")

## Part 6 — RAGAS Baseline Evaluation
# ── Node 10: RAGAS Evaluation Dataset ───────────────────────

import time

# ── Node 10: RAGAS Evaluation Dataset Construction ──────────

# Define the Mirror-Fact Questions and Ground Truths
RAGAS_QUESTIONS = [
    {
        "question": "List exactly the 5 Top Places for Odisha mentioned in the context.",
        "ground_truth": "The top places in Odisha are Jagannath Temple (Puri), Konark Sun Temple, Bhubaneswar, Chilika Lake, and Daringbadi."
    },
    {
        "question": "What is the TOTAL budget for 5 people for 10 days in Kerala?",
        "ground_truth": "The total budget is ₹287,500. This is calculated as ₹250,000 (₹5,000 per person per day) plus a 15% buffer of ₹37,500."
    },
    {
        "question": "What specific heritage accommodations are listed for Rajasthan?",
        "ground_truth": "The specific heritage accommodations listed for Rajasthan are Heritage Havelis and Palaces."
    },
    {
        "question": "Why is the daily budget for Northeast India states like Sikkim set at ₹7,000?",
        "ground_truth": "The budget is ₹7,000 per day because of the costs associated with SUV rentals and necessary travel permits."
    },
    {
        "question": "What are the recommended stays for Uttar Pradesh in the knowledge base?",
        "ground_truth": "The recommended stays for Uttar Pradesh are Riverside Ashrams and pilgrim guest houses."
    }
]

eval_dataset = []
print("🚀 Running agent for RAGAS evaluation...")

for i, rq in enumerate(RAGAS_QUESTIONS):
    # 1. Retrieval (Local operation, no rate limit)
    q_emb   = embedder.encode([rq["question"]]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=3)
    chunks  = results["documents"][0]
    
    # 2. Get Agent Answer with Exponential Backoff
    attempts = 0
    max_attempts = 3
    agent_answer = "Error: Model reached limit"
    
    while attempts < max_attempts:
        try:
            # unique thread_id for each eval question
            t_id = f"ragas_eval_{i}_{int(time.time())}"
            result = ask(rq["question"], thread_id=t_id)
            agent_answer = result.get("answer", "")
            break # Success!
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                attempts += 1
                wait_time = attempts * 15 # Wait 15s, then 30s
                print(f"  ⚠️ Rate Limit (429). Waiting {wait_time}s... (Attempt {attempts}/{max_attempts})")
                time.sleep(wait_time)
            else:
                print(f"  ❌ Critical Error: {error_msg}")
                break

    # 3. Store in Dataset
    eval_dataset.append({
        "question":     rq["question"],
        "answer":       agent_answer,
        "contexts":     chunks,
        "ground_truth": rq["ground_truth"]
    })
    
    print(f"  ✓ Processed {i+1}/{len(RAGAS_QUESTIONS)}: {rq['question'][:40]}...")
    
    # Add a small 2-second buffer between successful calls to avoid spiking TPM
    time.sleep(2)

print(f"\n✅ Eval dataset built: {len(eval_dataset)} rows")
# Run RAGAS (if installed) or fall back to manual scoring
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset

    ragas_data = Dataset.from_list(eval_dataset)
    print("Running RAGAS evaluation (1-2 minutes)...")

    ragas_result = evaluate(
        dataset=ragas_data,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    df = ragas_result.to_pandas()
    print("\n" + "=" * 45)
    print("BASELINE RAGAS SCORES")
    print("=" * 45)
    print(f"Faithfulness:      {df['faithfulness'].mean():.3f}")
    print(f"Answer Relevance:  {df['answer_relevancy'].mean():.3f}")
    print(f"Context Precision: {df['context_precision'].mean():.3f}")
    print("\n⚠️  Record these baseline scores. Re-run after any improvements.")

except ImportError:
    print("RAGAS not installed — running manual faithfulness scoring")
    faith_scores = []
    for row in eval_dataset:
        prompt = f"""Rate faithfulness 0.0-1.0. Reply with only a number.
Context: {row['contexts'][0][:300]}
Answer: {row['answer'][:200]}"""
        try:
            score = float(llm.invoke(prompt).content.strip().split()[0])
            score = max(0.0, min(1.0, score))
        except:
            score = 0.5
        faith_scores.append(score)
        print(f"  Q: {row['question'][:45]:45s} → {score:.2f}")

    avg = sum(faith_scores) / len(faith_scores)
    print(f"\nBaseline faithfulness: {avg:.3f}")
    print("Install RAGAS for full evaluation: pip install ragas datasets")

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
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ── 1. Configuration & State Definition ──────────────────
DOMAIN_NAME = "Bharat Travel Concierge"
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
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
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
        
        system_prompt = "You are the Bharat Travel Concierge. Answer strictly based on the provided context or tool results. If location info is missing, say you don't know."
        user_msg = f"Context: {context}\nTool Result: {tool}\nQuestion: {state['question']}"
        
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg)])
        return {"answer": response.content}

    def eval_node(state: CapstoneState):
        answer = state.get("answer", "")
        context = state.get("retrieved", "")[:3000] # Expanded context
        if not context or state.get("route") == "tool": return {"faithfulness": 1.0}
        
        prompt = f"Rate faithfulness (0.0 to 1.0). Does the answer use ONLY the context?\nContext: {context}\nAnswer: {answer}"
        result = llm.invoke(prompt).content
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
    workflow.add_node("tool", tool_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("eval", eval_node)
    workflow.add_node("save", save_node)

    workflow.set_entry_point("memory")
    workflow.add_edge("memory", "router")

    workflow.add_conditional_edges("router", lambda x: x["route"], {
        "retrieve": "retrieve",
        "tool": "tool",
        "skip": "answer"
    })
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
st.set_page_config(page_title=DOMAIN_NAME, page_icon="🤖", layout="centered")
st.title(f"🤖 {DOMAIN_NAME}")
st.caption(DOMAIN_DESCRIPTION)

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
    st.header("Session Info")
    st.write(f"**Thread ID:** `{st.session_state.thread_id}`")
    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()
    st.divider()
    st.write("**Supported States:** All 28 Indian States.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about destinations or trip budgets..."):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Consulting travel records..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            try:
                # Add small delay to prevent rapid-fire TPM limit hits
                time.sleep(1)
                result = agent_app.invoke({"question": prompt, "messages": st.session_state.messages[:-1]}, config=config)
                answer = result.get("answer", "I couldn't find a specific answer for that.")
                st.write(answer)
                
                faith = result.get("faithfulness", 0.0)
                sources = result.get("sources", [])
                if sources:
                    st.caption(f"🎯 Faithfulness: {faith:.2f} | 📚 Sources: {', '.join(sources)}")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                if "429" in str(e):
                    st.error("Rate limit reached. Please wait a moment before asking again.")
                else:
                    st.error(f"An error occurred: {e}")
