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
resume_agent = ResumeAIAgent("./MT24147_Yash choudhery.docx")
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
    query = request.messages[0] if request.messages else ""
    query.append("do not include (According to the, Yash Choudhery's resume ) line in response instead you can write Yash have skills ... yash has qualification ...." )

    allow_search = request.allow_search
    system_prompt = request.prompt
    provider = request.model_provider

    # NEW: Use resume agent if available and query is resume-related
    if resume_agent and resume_agent.resume_processor:
        from ai_agent import is_resume_related_query

        if is_resume_related_query(query):
            # Use RAG for resume questions
            response = get_response_from_ai_agent(
                llm_id, query, allow_search, system_prompt, provider,
                resume_processor=resume_agent.resume_processor  # Pass resume processor
            )
        else:
            # Original logic for non-resume questions
            response = get_response_from_ai_agent(
                llm_id, query, allow_search, system_prompt, provider
            )
    else:
        # Fallback to original if no resume available
        response = get_response_from_ai_agent(
            llm_id, , allow_search, system_prompt, provider
        )

    return response

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