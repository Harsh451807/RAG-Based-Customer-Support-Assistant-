import uuid
from workflow.state import GraphState
from retrieval.retriever import ContextRetriever
from llm.llm_client import LLMClient
from hitl.hitl_manager import HITLManager
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Node: Input Validation ──────────────────────────────────────────────
def input_node(state: GraphState) -> GraphState:
    """Validate and prepare the incoming user query."""
    query = state.get("user_query", "").strip()
    
    if not query:
        state["error"] = "Empty query received"
        state["final_answer"] = "Please enter a question. How can I help you?"
        state["next_action"] = "output"
        state["processing_complete"] = True
        return state
    
    # Truncate extremely long queries
    if len(query) > 1000:
        query = query[:1000]
        logger.warning("Query truncated to 1000 characters")
    
    # Initialize state defaults
    state["user_query"] = query
    state["session_id"] = state.get("session_id") or str(uuid.uuid4())[:8]
    state["conversation_history"] = state.get("conversation_history", [])
    state["retrieved_chunks"] = []
    state["escalation_triggered"] = False
    state["sources"] = []
    state["next_action"] = "classify"
    
    logger.info(f"[Session: {state['session_id']}] Processing query: {query[:80]}")
    return state


# ─── Node: Intent Classification ─────────────────────────────────────────
def classify_node(state: GraphState) -> GraphState:
    """Classify the intent of the user query."""
    query = state["user_query"].lower()
    
    # ── Greeting Detection ─────────────────────────────────────────────
    greetings = {"hello", "hi", "hey", "good morning", "good afternoon", 
                 "good evening", "howdy", "greetings", "what's up", "sup"}
    words = set(query.split())
    
    if words.intersection(greetings) and len(query.split()) <= 5:
        state["intent"] = "GREETING"
        logger.info(f"Intent: GREETING")
        return state
    
    # ── Explicit Escalation Detection ──────────────────────────────────
    escalation_phrases = [
        "speak to human", "talk to agent", "real person", "human agent",
        "escalate this", "not satisfied", "want to complain", "supervisor",
        "manager please", "human please", "real agent", "speak to someone"
    ]
    if any(phrase in query for phrase in escalation_phrases):
        state["intent"] = "ESCALATE"
        state["escalation_reason"] = "Customer explicitly requested human agent"
        logger.info("Intent: ESCALATE (explicit request)")
        return state
    
    # ── Complexity Detection ────────────────────────────────────────────
    if query.count("?") > 2:
        state["intent"] = "COMPLEX"
        state["escalation_reason"] = "Multi-part query detected"
        logger.info("Intent: COMPLEX (multiple questions)")
        return state
    
    # ── Default: Knowledge Query ────────────────────────────────────────
    state["intent"] = "KNOWLEDGE_QUERY"
    logger.info("Intent: KNOWLEDGE_QUERY")
    return state


# ─── Node: Direct Response (Greetings) ───────────────────────────────────
def direct_response_node(state: GraphState) -> GraphState:
    """Handle simple greetings without RAG."""
    state["final_answer"] = (
        "Hello! 👋 I'm your customer support assistant. "
        "I can help you with questions about our products, policies, "
        "and services. What can I help you with today?"
    )
    state["sources"] = []
    state["next_action"] = "output"
    return state


# ─── Node: Retrieval ──────────────────────────────────────────────────────
def retrieve_node(state: GraphState, retriever: ContextRetriever) -> GraphState:
    """Retrieve relevant document chunks from ChromaDB."""
    result = retriever.retrieve(state["user_query"])
    
    state["retrieved_chunks"] = result.chunks
    state["retrieval_confidence"] = result.top_score
    state["formatted_context"] = result.context_text
    
    if result.is_confident:
        state["next_action"] = "generate"
        logger.info(f"Retrieval confident (score: {result.top_score:.3f})")
    else:
        state["next_action"] = "hitl"
        state["escalation_reason"] = (
            f"No relevant documentation found "
            f"(confidence: {result.top_score:.3f}, "
            f"threshold: {Config.MIN_SIMILARITY_SCORE})"
        )
        logger.warning(f"Retrieval below threshold: {result.top_score:.3f}")
    
    return state


# ─── Node: LLM Generation ────────────────────────────────────────────────
def generate_node(state: GraphState, llm_client: LLMClient) -> GraphState:
    """Generate answer using LLM with retrieved context."""
    
    # Build conversation history string
    history_str = ""
    recent_history = state["conversation_history"][-4:]  # Last 2 turns
    for turn in recent_history:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        history_str += f"{role.upper()}: {content}\n"
    
    # Construct the prompt
    prompt = f"""{Config.SYSTEM_PROMPT}

RELEVANT CONTEXT FROM KNOWLEDGE BASE:
{state['formatted_context']}

CONVERSATION HISTORY:
{history_str if history_str else "This is the start of the conversation."}

CUSTOMER QUESTION: {state['user_query']}

ANSWER:"""
    
    response = llm_client.complete(prompt)
    
    state["llm_answer"] = response.get("answer", "")
    state["llm_confidence"] = response.get("confidence", 0.0)
    
    logger.info(f"LLM response generated ({len(state['llm_answer'].split())} words)")
    return state


# ─── Node: Confidence Evaluation ─────────────────────────────────────────
def confidence_node(state: GraphState) -> GraphState:
    """Evaluate LLM response quality and route accordingly."""
    answer = state.get("llm_answer", "")
    
    # Phrases that indicate the LLM couldn't answer
    uncertainty_phrases = [
        "i don't have enough information",
        "i'm not sure",
        "i cannot determine",
        "insufficient information",
        "i'm unable to answer",
        "not mentioned in the context",
        "i don't know"
    ]
    
    has_uncertainty = any(
        phrase in answer.lower() 
        for phrase in uncertainty_phrases
    )
    
    is_too_short = len(answer.split()) < 8
    
    if has_uncertainty or is_too_short:
        state["next_action"] = "hitl"
        state["escalation_reason"] = (
            "AI system expressed uncertainty - "
            "routing to human agent for accurate response"
        )
        logger.info("Confidence check FAILED - escalating to HITL")
    else:
        state["final_answer"] = answer
        state["next_action"] = "output"
        logger.info("Confidence check PASSED - serving AI response")
    
    return state


# ─── Node: HITL Escalation ────────────────────────────────────────────────
def hitl_node(state: GraphState, hitl_manager: HITLManager) -> GraphState:
    """Handle escalation to human agent."""
    ticket_id = hitl_manager.create_ticket(
        session_id=state["session_id"],
        user_query=state["user_query"],
        escalation_reason=state.get("escalation_reason", "Escalation triggered"),
        conversation_history=state["conversation_history"],
        ai_attempt=state.get("llm_answer", ""),
        retrieved_context=state.get("formatted_context", "")
    )
    
    state["escalation_triggered"] = True
    state["ticket_id"] = ticket_id
    
    # Get human response (CLI simulation)
    human_response = hitl_manager.get_human_response(
        ticket_id=ticket_id,
        user_query=state["user_query"],
        escalation_reason=state.get("escalation_reason", "")
    )
    
    state["human_response"] = human_response
    state["final_answer"] = (
        f"🧑‍💼 **Response from Human Agent:**\n\n"
        f"{human_response}\n\n"
        f"📋 Reference: {ticket_id}"
    )
    state["next_action"] = "output"
    
    logger.info(f"HITL complete. Ticket: {ticket_id}")
    return state


# ─── Node: Output Formatting ──────────────────────────────────────────────
def output_node(state: GraphState) -> GraphState:
    """Format final response and update conversation history."""
    
    # Collect source citations
    sources = list(set([
        f"{chunk.source} (Page {chunk.page})"
        for chunk in state.get("retrieved_chunks", [])
    ]))
    state["sources"] = sources
    
    # Update conversation history
    state["conversation_history"].append({
        "role": "user",
        "content": state["user_query"]
    })
    state["conversation_history"].append({
        "role": "assistant",
        "content": state["final_answer"],
        "sources": sources,
        "escalated": state.get("escalation_triggered", False),
        "ticket_id": state.get("ticket_id")
    })
    
    state["processing_complete"] = True
    logger.info("Response ready for delivery")
    return state