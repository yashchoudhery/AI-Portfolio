from pydantic import BaseModel,Field
from ai_agent import get_response_from_ai_agent, ResumeAIAgent

from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request

# step 1

from typing import List, Optional

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    prompt: str
    messages: List[str] = Field(default_factory=list)
    allow_search: bool = False


from fastapi import FastAPI

allowed_model_names=  ["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile", "gpt-4o-mini"]

app = FastAPI(title="AI Portfolio")

# NEW: Initialize Resume AI Agent once when server starts
print("🚀 Initializing Resume AI Agent...")
resume_agent = ResumeAIAgent("./resume.txt")
print("✅ Resume Agent ready!")


@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API endpoint for chat with AI Portfolio
    Now supports RAG for resume questions!
    """
    if request.model_name not in allowed_model_names:
        raise HTTPException(status_code=400, detail="Invalid model name")

    llm_id = request.model_name
    original_query = request.messages[0] if request.messages else ""
    allow_search = request.allow_search
    provider = request.model_provider

    # Add instructions to make AI respond in first person (as Yash)
    instruction = """
    Important: Respond as if you ARE Yash Choudhery answering in first person.
    Do NOT write phrases like:
    - "According to Yash Choudhery's resume..."
    - "According to the provided information..."
    - "Yash's qualifications are..."
    - "His skills include..."

    Instead, write in first person:
    - "My qualifications are..."
    - "I have skills in..."
    - "I worked on..."
    - "My projects include..."

    Respond naturally as Yash speaking about himself.
    """

    # Combine instruction with original query
    query = f"{instruction}\n\nUser question: {original_query}"

    # System prompt for first-person responses
    # Strong grounding + first-person + brevity + safety
    grounding_instruction = (
        "You are Yash Choudhery. Answer in the FIRST PERSON (use 'I', 'my', etc.). "
        "STRICTLY use only information present in the provided resume context. "
        "Do NOT invent, infer, or add facts not present in the resume. "
        "If the requested fact is NOT in the resume, reply exactly: \"I don't have that information.\". "
        "Keep the answer concise (max 120 words). "
        "When you quote a fact from the resume, keep it brief and factual — do not add extra context."
    )

    # Final system prompt sent to the model
    system_prompt = grounding_instruction

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # NEW: Stronger resume checking
    from ai_agent import is_resume_related_query

    is_resume_q = is_resume_related_query(query)
    print(f"🔍 Query: '{query[:50]}...'")
    print(f"📋 Is Resume Related: {is_resume_q}")

    # If NOT resume related, return redirect message immediately
    if not is_resume_q:
        return {
            "response": "I can only answer questions about Yash Choudhery's professional background, skills, experience, and qualifications. Please ask about:\n\n• Work experience and roles\n• Technical skills and programming languages\n• Educational background\n• Projects and achievements\n• Professional qualifications",
            "is_resume_related": False
        }

    # If IS resume related, use RAG
    if resume_agent and resume_agent.resume_processor:
        response = get_response_from_ai_agent(
            llm_id, query, False, system_prompt, provider,
            resume_processor=resume_agent.resume_processor
        )
    else:
        response = get_response_from_ai_agent(
            llm_id, query, allow_search, system_prompt, provider
        )

    return {
        "response": response,
        "is_resume_related": True
    }


@app.get("/health")
def health_check():
    """Check if server and resume agent are ready"""
    return {
        "status": "healthy",
        "resume_loaded": resume_agent.resume_processor is not None if resume_agent else False
    }

@app.get("/info")
def get_info():
    """Get information about available capabilities"""
    return {
        "name": "AI Portfolio Chat",
        "models": allowed_model_names,
        "resume_qa_enabled": resume_agent.resume_processor is not None if resume_agent else False,
        "example_questions": [
            "What programming languages do you know?",
            "Tell me about your work experience",
            "What's your educational background?"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)