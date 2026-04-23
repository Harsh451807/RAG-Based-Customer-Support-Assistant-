import os
import time
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

from ingestion.document_loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embedder import EmbeddingEngine
from storage.vector_store import VectorStoreManager
from retrieval.retriever import ContextRetriever
from llm.llm_client import LLMClient
from hitl.hitl_manager import HITLManager
from workflow.graph import build_workflow
from config import Config

console = Console()

os.environ["LANGGRAPH_STRICT_MSGPACK"] = "false"

def initialize_system() -> dict:
    """Initialize all system components."""
    console.print(Panel(
        "[bold blue]RAG Customer Support Assistant[/bold blue]\n"
        "Initializing system components...",
        border_style="blue"
    ))
    
    # Initialize components
    embedder = EmbeddingEngine()
    vector_store = VectorStoreManager()
    
    # Ingest documents if collection is empty
    if vector_store.get_count() == 0:
        console.print("[yellow]No documents in database. Running ingestion...[/yellow]")
        ingest_documents(embedder, vector_store)
    else:
        console.print(f"[green]✓[/green] Knowledge base loaded ({vector_store.get_count()} chunks)")
    
    retriever = ContextRetriever(embedder, vector_store)
    llm_client = LLMClient()
    hitl_manager = HITLManager()
    
    # Build workflow graph
    app = build_workflow(retriever, llm_client, hitl_manager)
    
    console.print("[green]✓ System ready![/green]\n")
    
    return {
        "app": app,
        "hitl_manager": hitl_manager
    }

def ingest_documents(embedder: EmbeddingEngine, vector_store: VectorStoreManager):
    """Run the document ingestion pipeline."""
    loader = DocumentLoader()
    chunker = DocumentChunker()
    
    documents = loader.load_directory(Config.DOCUMENTS_DIR)
    
    if not documents:
        console.print(
            f"[red]No PDFs found in {Config.DOCUMENTS_DIR}[/red]\n"
            "Please add PDF files to the documents directory."
        )
        return
    
    chunks = chunker.chunk_documents(documents)
    embedded_chunks = embedder.embed_chunks(chunks)
    vector_store.store_embeddings(embedded_chunks)
    
    console.print(f"[green]✓ Ingested {len(chunks)} chunks from {len(documents)} pages[/green]")

def run_query(app, query: str, session_id: str, history: list) -> dict:
    """Execute a single query through the workflow."""
    initial_state = {
        "user_query": query,
        "session_id": session_id,
        "conversation_history": history,
        "intent": "",
        "retrieved_chunks": [],
        "retrieval_confidence": 0.0,
        "formatted_context": "",
        "llm_answer": "",
        "llm_confidence": 0.0,
        "final_answer": "",
        "sources": [],
        "escalation_triggered": False,
        "escalation_reason": "",
        "ticket_id": None,
        "next_action": "",
        "error": None,
        "processing_complete": False
    }
    
    config = {"configurable": {"thread_id": session_id}}
    
    start_time = time.time()
    result = app.invoke(initial_state, config=config)
    elapsed = round((time.time() - start_time) * 1000)
    
    return {
        "answer": result.get("final_answer", "No response generated"),
        "sources": result.get("sources", []),
        "escalated": result.get("escalation_triggered", False),
        "ticket_id": result.get("ticket_id"),
        "confidence": result.get("retrieval_confidence", 0.0),
        "history": result.get("conversation_history", []),
        "elapsed_ms": elapsed
    }

def display_response(response: dict):
    """Display the response in a formatted way."""
    answer_text = Text(response["answer"])
    
    if response["escalated"]:
        panel_color = "yellow"
        title = "🧑‍💼 Human Agent Response"
    else:
        panel_color = "green"
        title = "🤖 Assistant"
    
    console.print(Panel(answer_text, title=title, border_style=panel_color))
    
    if response["sources"]:
        console.print(f"[dim]📚 Sources: {', '.join(response['sources'])}[/dim]")
    
    console.print(
        f"[dim]⏱ {response['elapsed_ms']}ms | "
        f"Confidence: {response['confidence']:.2f} | "
        f"Escalated: {'Yes' if response['escalated'] else 'No'}[/dim]\n"
    )

def main():
    """Main CLI interaction loop."""
    system = initialize_system()
    app = system["app"]
    
    session_id = "session_001"
    conversation_history = []
    
    console.print(
        "[bold]Customer Support Assistant[/bold]\n"
        "Commands: 'quit' to exit | 'stats' for ticket stats | 'clear' to reset session\n"
    )
    
    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            console.print("[yellow]Thank you for using Customer Support. Goodbye![/yellow]")
            break
        
        if user_input.lower() == "clear":
            conversation_history = []
            console.print("[green]Session cleared.[/green]")
            continue
        
        if user_input.lower() == "stats":
            stats = system["hitl_manager"].get_ticket_stats()
            rprint(stats)
            continue
        
        # Process query
        response = run_query(app, user_input, session_id, conversation_history)
        conversation_history = response["history"]
        display_response(response)

if __name__ == "__main__":
    main()