from functools import partial
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from workflow.state import GraphState
from workflow.nodes import (
    input_node, classify_node, direct_response_node,
    retrieve_node, generate_node, confidence_node,
    hitl_node, output_node
)
from workflow.router import (
    intent_router, retrieval_router, confidence_router
)
from retrieval.retriever import ContextRetriever
from llm.llm_client import LLMClient
from hitl.hitl_manager import HITLManager
from utils.logger import get_logger

logger = get_logger(__name__)

def build_workflow(
    retriever: ContextRetriever,
    llm_client: LLMClient,
    hitl_manager: HITLManager
):
    """Build and compile the LangGraph workflow."""
    
    workflow = StateGraph(GraphState)
    
    # ── Register Nodes ────────────────────────────────────────────────────
    workflow.add_node("input_node", input_node)
    workflow.add_node("classify_node", classify_node)
    workflow.add_node("direct_response_node", direct_response_node)
    workflow.add_node("retrieve_node", partial(retrieve_node, retriever=retriever))
    workflow.add_node("generate_node", partial(generate_node, llm_client=llm_client))
    workflow.add_node("confidence_node", confidence_node)
    workflow.add_node("hitl_node", partial(hitl_node, hitl_manager=hitl_manager))
    workflow.add_node("output_node", output_node)
    
    # ── Define Entry Point ────────────────────────────────────────────────
    workflow.set_entry_point("input_node")
    
    # ── Define Edges ──────────────────────────────────────────────────────
    # input → error check or classify
    workflow.add_conditional_edges(
        "input_node",
        lambda s: "output_node" if s.get("processing_complete") else "classify_node",
        {"output_node": "output_node", "classify_node": "classify_node"}
    )
    
    # classify → route by intent
    workflow.add_conditional_edges(
        "classify_node",
        intent_router,
        {
            "direct_response_node": "direct_response_node",
            "hitl_node": "hitl_node",
            "retrieve_node": "retrieve_node"
        }
    )
    
    # retrieve → route by confidence
    workflow.add_conditional_edges(
        "retrieve_node",
        retrieval_router,
        {
            "generate": "generate_node",
            "hitl": "hitl_node"
        }
    )
    
    # generate → confidence check
    workflow.add_edge("generate_node", "confidence_node")
    
    # confidence → route by quality
    workflow.add_conditional_edges(
        "confidence_node",
        confidence_router,
        {
            "output": "output_node",
            "hitl": "hitl_node"
        }
    )
    
    # All paths converge at output
    workflow.add_edge("direct_response_node", "output_node")
    workflow.add_edge("hitl_node", "output_node")
    workflow.add_edge("output_node", END)
    
    # ── Compile with Memory Checkpointing ────────────────────────────────
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    logger.info("LangGraph workflow compiled successfully")
    return app