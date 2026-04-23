from typing import TypedDict, List, Optional, Any
from retrieval.retriever import RetrievedChunk

class GraphState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────
    user_query: str
    session_id: str
    conversation_history: List[dict]
    
    # ── Classification ──────────────────────────────────────────────
    intent: str
    
    # ── Retrieval ───────────────────────────────────────────────────
    retrieved_chunks: List[RetrievedChunk]
    retrieval_confidence: float
    formatted_context: str
    
    # ── Generation ──────────────────────────────────────────────────
    llm_answer: str
    llm_confidence: float
    has_uncertainty: bool
    
    # ── Output ──────────────────────────────────────────────────────
    final_answer: str
    sources: List[str]
    
    # ── HITL ────────────────────────────────────────────────────────
    escalation_triggered: bool
    escalation_reason: str
    ticket_id: Optional[str]
    human_response: Optional[str]
    
    # ── Control ─────────────────────────────────────────────────────
    next_action: str
    error: Optional[str]
    processing_complete: bool