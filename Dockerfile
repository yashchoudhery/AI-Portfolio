FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY Backend.py ai_agent.py resume.txt ./

# Expose port 7860 (HF Spaces default)
ENV PORT=7860

# Run the application
CMD ["uvicorn", "Backend:app", "--host", "0.0.0.0", "--port", "7860"]
