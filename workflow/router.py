from workflow.state import GraphState

def intent_router(state: GraphState) -> str:
    """Route based on classified intent."""
    intent = state.get("intent", "KNOWLEDGE_QUERY")
    
    routes = {
        "GREETING": "direct_response_node",
        "ESCALATE": "hitl_node",
        "COMPLEX": "hitl_node",
        "KNOWLEDGE_QUERY": "retrieve_node"
    }
    
    return routes.get(intent, "retrieve_node")


def retrieval_router(state: GraphState) -> str:
    """Route based on retrieval confidence."""
    return state.get("next_action", "hitl_node")


def confidence_router(state: GraphState) -> str:
    """Route based on LLM confidence."""
    return state.get("next_action", "hitl_node")


def output_error_router(state: GraphState) -> str:
    """Handle error cases."""
    if state.get("error"):
        return "output_node"
    return state.get("next_action", "classify_node")