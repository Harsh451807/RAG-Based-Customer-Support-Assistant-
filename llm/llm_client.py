# llm/llm_client.py

from groq import Groq
from config import Config        # ✅ import Config class

class LLMClient:
    
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)  # ✅ Config.GROQ_API_KEY
        self.model = Config.LLM_MODEL                     # ✅ Config.LLM_MODEL
        print(f"  LLM ready. Model: {self.model}")
    
    def complete(self, prompt: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=Config.LLM_TEMPERATURE,       # ✅ Config.LLM_TEMPERATURE
                max_tokens=Config.LLM_MAX_TOKENS          # ✅ Config.LLM_MAX_TOKENS
            )
            
            answer = response.choices[0].message.content.strip()
            
            uncertainty_keywords = [
                "i don't have enough information",
                "i don't know",
                "not sure",
                "unclear",
                "insufficient",
                "unable to answer",
                "not mentioned in the context"
            ]
            
            has_uncertainty = any(
                kw in answer.lower()
                for kw in uncertainty_keywords
            )
            
            confidence = 0.3 if has_uncertainty else 0.85
            
            return {
                "answer": answer,
                "confidence": confidence,
                "tokens_used": response.usage.total_tokens,
                "model": self.model
            }
        
        except Exception as e:
            print(f"  LLM Error: {e}")
            return {
                "answer": "I don't have enough information to answer this question accurately.",
                "confidence": 0.0,
                "tokens_used": 0,
                "error": str(e)
            }