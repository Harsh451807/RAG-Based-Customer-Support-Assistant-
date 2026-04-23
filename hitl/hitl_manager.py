import json
import uuid
import os
from datetime import datetime
from typing import List, Optional
from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

class HITLManager:
    
    def __init__(self):
        self.tickets_file = Config.TICKETS_FILE
        os.makedirs(os.path.dirname(self.tickets_file), exist_ok=True)
        
        # Load or initialize tickets store
        if os.path.exists(self.tickets_file):
            with open(self.tickets_file, 'r') as f:
                self.tickets = json.load(f)
        else:
            self.tickets = {}
    
    def create_ticket(
        self,
        session_id: str,
        user_query: str,
        escalation_reason: str,
        conversation_history: List[dict],
        ai_attempt: str = "",
        retrieved_context: str = ""
    ) -> str:
        """Create an escalation ticket and return ticket ID."""
        
        ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:4].upper()}"
        
        ticket = {
            "ticket_id": ticket_id,
            "session_id": session_id,
            "status": "OPEN",
            "priority": self._calculate_priority(escalation_reason),
            "user_query": user_query,
            "escalation_reason": escalation_reason,
            "conversation_history": conversation_history[-6:],  # Last 3 turns
            "ai_attempt": ai_attempt,
            "retrieved_context": retrieved_context[:500] if retrieved_context else "",
            "created_at": datetime.now().isoformat(),
            "resolved_at": None,
            "human_response": None,
            "assigned_agent": None
        }
        
        self.tickets[ticket_id] = ticket
        self._save_tickets()
        
        logger.info(f"Ticket created: {ticket_id} | Priority: {ticket['priority']}")
        return ticket_id
    
    def get_human_response(
        self,
        ticket_id: str,
        user_query: str,
        escalation_reason: str
    ) -> str:
        """
        Simulates waiting for human agent response.
        In production: polls ticket system or uses webhook.
        In demo: CLI prompt for agent input.
        """
        print("\n" + "="*65)
        print(f"🚨  ESCALATION REQUIRED  |  Ticket: {ticket_id}")
        print("="*65)
        print(f"📧 Customer Query: {user_query}")
        print(f"📌 Reason: {escalation_reason}")
        
        if self.tickets[ticket_id].get("ai_attempt"):
            print(f"🤖 AI Attempted: {self.tickets[ticket_id]['ai_attempt'][:200]}...")
        
        print("-"*65)
        
        # Get human response
        response = input("🧑‍💼 Agent Response: ").strip()
        
        if not response:
            response = (
                "Thank you for contacting us. A support agent will "
                "follow up with you shortly via email."
            )
        
        # Update ticket
        self.tickets[ticket_id]["status"] = "RESOLVED"
        self.tickets[ticket_id]["human_response"] = response
        self.tickets[ticket_id]["resolved_at"] = datetime.now().isoformat()
        self._save_tickets()
        
        print("="*65 + "\n")
        logger.info(f"Ticket {ticket_id} resolved by human agent")
        
        return response
    
    def _calculate_priority(self, reason: str) -> str:
        """Determine ticket priority based on escalation reason."""
        if "explicitly requested" in reason.lower():
            return "HIGH"
        elif "api" in reason.lower() or "error" in reason.lower():
            return "HIGH"
        elif "uncertainty" in reason.lower():
            return "MEDIUM"
        else:
            return "LOW"
    
    def _save_tickets(self) -> None:
        """Persist tickets to JSON file."""
        with open(self.tickets_file, 'w') as f:
            json.dump(self.tickets, f, indent=2)
    
    def get_ticket_stats(self) -> dict:
        """Return summary statistics for all tickets."""
        total = len(self.tickets)
        if total == 0:
            return {"total": 0}
        
        statuses = [t["status"] for t in self.tickets.values()]
        priorities = [t["priority"] for t in self.tickets.values()]
        
        return {
            "total": total,
            "open": statuses.count("OPEN"),
            "resolved": statuses.count("RESOLVED"),
            "high_priority": priorities.count("HIGH"),
            "medium_priority": priorities.count("MEDIUM"),
            "low_priority": priorities.count("LOW")
        }