import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent

from langchain_core.messages.ai import AIMessage
import inspect
# NEW: Add these imports for RAG functionality
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)
#   step1
GROQ_API_KEY = os.environ['GROQ_API_KEY']
TAVILY_API_KEY=os.environ['TAVILY_API_KEY']
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']


# NEW: Resume configuration
RESUME_PDF_PATH = "./resume.txt"  # ← CHANGE THIS TO YOUR ACTUAL RESUME PATH


# step 2
openai_llm=ChatOpenAI(model="gpt-4o-mini")
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")
groq_llm2=ChatGroq(model="mixtral-8x7b-32768")
try:

    from langchain_tavily import TavilySearch as TavilySearchTool
except Exception:
    try:
        # fallback for older langchain versions (deprecated)
        from langchain import TavilySearchResults as TavilySearchTool
    except Exception as e:
        raise ImportError(
            "Could not import TavilySearch. Install with: pip install -U langchain-tavily"
        ) from e




# step  3
# which is personalised and able to answer all the questions related to my resume
system_prompt = "Act as an AI chatbot which is personalised for and able to answer all the questions related to my resume, you can get the information of yash from given linkedin"


# NEW: Resume Processor Class
class ResumeProcessor:
    def __init__(self, resume_path):
        self.resume_path = resume_path
        self.vector_store = None
        self.retriever = None

        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def setup_vector_store(self):
        try:
            if not os.path.exists(self.resume_path):
                print(f"Resume not found at: {self.resume_path}")
                return False

            print(f"Processing resume: {self.resume_path}")

            # Smart loader - automatically detect file type
            if self.resume_path.endswith('.docx') or self.resume_path.endswith('.doc'):
                print("Loading Word document...")
                loader = Docx2txtLoader(self.resume_path)
            elif self.resume_path.endswith('.pdf'):
                print("Loading PDF...")
                loader = PyPDFLoader(self.resume_path)
            elif self.resume_path.endswith('.txt'):
                print("Loading text file...")
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(self.resume_path)
            else:
                raise ValueError("Resume must be .pdf, .doc, .docx, or .txt file")

            documents = loader.load()
            print(f"\n📄 Loaded {len(documents)} document(s)")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # INCREASED from 500 to 1000
                chunk_overlap=150,  # INCREASED from 50 to 150
                separators=[  # IMPROVED: Resume-specific separators
                    "\n\n\n",  # Multiple line breaks (section breaks)
                    "\n\n",  # Double line breaks
                    "\n---",  # Divider lines
                    "---",  # Simple dividers
                    "\n•",  # Bullet points
                    "\n-",  # Dash points
                    "\n*",  # Asterisk points
                    "\n",  # Single line breaks
                    ". ",  # Sentence breaks
                    " ",  # Word breaks
                    ""  # Character level
                ]
            )

            chunks = text_splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks")

            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 7}  # INCREASED from 3 to 7
            )

            print("Resume processing complete!")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def get_relevant_context(self, query, k=7):
        """Get relevant context with enhanced formatting"""
        if not self.retriever:
            return ""
        try:
            # IMPROVED: Get more documents for comprehensive context
            docs = self.retriever.get_relevant_documents(query)
            context_parts = []

            for i, doc in enumerate(docs[:k]):
                # Add section markers for better context organization
                context_parts.append(f"--- Context Section {i + 1} ---\n{doc.page_content.strip()}\n")

            return "\n".join(context_parts)
        except Exception as e:
            print(f"❌ Error retrieving context: {e}")
            return ""


def is_resume_related_query(query):
    """
    Enhanced checking for resume-related queries.
    Returns True for professional/career questions with much better coverage.
    """
    query_lower = query.lower()

    # GREATLY EXPANDED: More comprehensive resume keywords
    resume_keywords = [
        # Personal/Background
        "yash", "choudhery", "your", "you", "tell me about", "about",
        "who are you", "about yourself", "introduce yourself", "background", "profile",

        # Career/Work - EXPANDED significantly
        "experience", "work", "job", "career", "role", "position", "employment",
        "responsibility", "worked", "working", "professional", "industry",
        "company", "organization", "employer", "wipro", "persistent", "infosys",
        "researcher", "engineer", "developer", "intern", "internship",

        # Skills/Technical - EXPANDED with specific technologies from resume
        "skills", "technical", "programming", "language", "framework", "technology",
        "tool", "software", "expertise", "proficiency", "know", "familiar",
        "python", "java", "javascript", "c++", "pytorch", "tensorflow", "scikit-learn",
        "machine learning", "ai", "artificial intelligence", "deep learning", "nlp",
        "ml", "data science", "angular", "react", "spring boot", "mysql", "mongodb",
        "docker", "git", "github", "cuda", "linux", "numpy", "pandas", "matplotlib",

        # Projects - IMPROVED with specific project names and types
        "project", "projects", "built", "created", "developed", "implementation",
        "development", "coding", "programming", "application", "system", "solution",
        "vfl", "federated learning", "speech", "keyword spotting", "speaker verification",
        "tracking", "object tracking", "conferencing", "video conferencing",
        "wifi", "simulator", "healthcare", "web application", "geo-spatial",

        # Education - EXPANDED with specific institutions
        "education", "degree", "university", "college", "school", "iiit", "delhi",
        "study", "studied", "certification", "qualified", "mtech", "btech", "diploma",
        "cgpa", "percentage", "gate", "akhilesh das gupta", "tool engineering",

        # Achievements/Awards - EXPANDED
        "achievement", "accomplishment", "portfolio", "award", "recognition",
        "qualification", "certificate", "gate", "spot award", "qualified"
    ]

    # Check if ANY keyword matches
    has_resume_keyword = any(keyword in query_lower for keyword in resume_keywords)

    # Exclude common non-resume questions (blocklist)
    non_resume_patterns = [
        "weather", "time", "date", "news", "stock", "price",
        "joke", "story", "recipe", "cook", "food",
        "movie", "song", "game", "sports", "politics",
        "calculate", "math", "solve", "translate","tourism", "place", "location", "destination",

        # General Knowledge
        "weather", "temperature", "climate", "forecast", "rain", "sunny", "cloudy",
        "time", "date", "today", "tomorrow", "yesterday", "current time", "clock",
        "news", "latest", "breaking", "headlines", "current events", "politics",

        # Entertainment & Media
        "movie", "film", "cinema", "actor", "actress", "celebrity", "star",
        "health", "medical", "doctor", "hospital", "medicine", "sick",

        "london", "paris", "new york", "tokyo", "how was london", "visit london",

        # Weather (specific)
        "weather", "temperature", "rain", "sunny", "what's the weather",

        # Entertainment (specific)
        "joke", "tell me a joke", "movie recommendation", "song recommendation",

        # Services (specific)
        "what time is it", "current time", "latest news", "stock price",
        "book ticket", "hotel booking", "recipe", "how to cook"
    ]

    has_non_resume = any(pattern in query_lower for pattern in non_resume_patterns)

    # Return True ONLY if has resume keyword AND doesn't have non-resume patterns
    return has_resume_keyword and not has_non_resume


def get_query_type(query):
    """Classify the type of resume query for specialized handling"""
    query_lower = query.lower()

    # More specific project detection
    if any(word in query_lower for word in [
        "project", "projects", "built", "created", "developed", "application",
        "system", "solution", "implementation", "vfl", "federated learning",
        "tracking", "conferencing", "simulator"
    ]):
        return "projects"

    # More specific skills detection
    elif any(word in query_lower for word in [
        "skill", "skills", "technical", "programming", "technology", "framework",
        "language", "tool", "python", "java", "pytorch", "tensorflow", "proficiency"
    ]):
        return "skills"

    # Work experience detection
    elif any(word in query_lower for word in [
        "experience", "work", "job", "career", "employment", "role", "position",
        "wipro", "persistent", "infosys", "researcher", "engineer"
    ]):
        return "experience"

    # Education detection
    elif any(word in query_lower for word in [
        "education", "degree", "university", "study", "iiit", "college",
        "qualification", "mtech", "btech", "cgpa"
    ]):
        return "education"
    else:
        return "general"


def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider, resume_processor=None):
    # Your existing LLM setup (unchanged)
    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id)
    elif provider == "Groq2":
        llm = ChatGroq(model=llm_id)

    # NEW: Use RAG if resume processor provided and query is resume-related
    if resume_processor and is_resume_related_query(query):
        print("🤖 Using Enhanced RAG...")

        # Get query type for specialized handling
        query_type = get_query_type(query)
        print(f"🎯 Query type detected: {query_type}")

        context = resume_processor.get_relevant_context(query, k=7)

        if context:
            # IMPROVED: Query-type specific prompts for better responses
            if query_type == "skills":
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are Yash Choudhery answering questions about your technical skills and expertise.

                    CRITICAL: Always respond in FIRST PERSON using 'I', 'my', 'me'.

                    For skills questions, provide comprehensive details:
                    1. List specific skills you possess with proficiency levels
                    2. Mention which projects used which skills (correlate skills with projects)
                    3. Include programming languages, frameworks, tools, technologies
                    4. Mention specific experience and duration where relevant

                    Be thorough and provide examples from your experience.

                    Context from my background:
                    {context}"""),
                    ("human", "{input}")
                ])

            elif query_type == "projects":
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are Yash Choudhery answering questions about your projects and technical work.

                    CRITICAL: Always respond in FIRST PERSON using 'I', 'my', 'me'.

                    For project questions, provide comprehensive details:
                    1. Project names, descriptions, and objectives
                    2. Technologies, frameworks, and skills used in each project
                    3. My specific contributions, role, and achievements
                    4. Technical challenges solved and impact/results
                    5. Duration, team size, and collaboration details

                    Be thorough and technical in your explanations.

                    Context from my background:
                    {context}"""),
                    ("human", "{input}")
                ])

            elif query_type == "experience":
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are Yash Choudhery answering questions about your work experience and career.

                    CRITICAL: Always respond in FIRST PERSON using 'I', 'my', 'me'.

                    For experience questions, provide:
                    1. Detailed work history with companies, roles, and durations
                    2. Key responsibilities and achievements at each position
                    3. Technologies used and skills developed
                    4. Impact and contributions to projects/teams
                    5. Career progression and growth

                    Context from my background:
                    {context}"""),
                    ("human", "{input}")
                ])

            elif query_type == "education":
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are Yash Choudhery answering questions about your educational background.

                    CRITICAL: Always respond in FIRST PERSON using 'I', 'my', 'me'.

                    For education questions, provide:
                    1. Detailed academic qualifications with institutions
                    2. Degrees, specializations, and academic performance
                    3. Relevant coursework and technical subjects
                    4. Academic achievements and certifications
                    5. How education relates to career and skills

                    Context from my background:
                    {context}"""),
                    ("human", "{input}")
                ])
            else:
                # General resume prompt - enhanced
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are Yash Choudhery answering questions about yourself and your professional background.

                    CRITICAL: Always respond in FIRST PERSON using 'I', 'my', 'me'.
                    Never say "According to the resume" or "Yash has" or "His skills".

                    Provide comprehensive, detailed responses covering relevant aspects of your background.

                    Examples of correct responses:
                    - "I have experience in Python and Java..."
                    - "My qualifications include..."
                    - "I worked on a project where..."
                    - "I am proficient in..."

                    Context from my background:
                    {context}"""),
                    ("human", "{input}")
                ])

            document_chain = create_stuff_documents_chain(llm, rag_prompt)
            rag_chain = create_retrieval_chain(resume_processor.retriever, document_chain)
            response = rag_chain.invoke({"input": query})

            print(f"✅ Generated {query_type}-specific response")
            return response["answer"]

    # NEW: Redirect non-resume questions
    if resume_processor and not is_resume_related_query(query):
        return "I can only answer questions about my professional background and resume."

    # Your ORIGINAL logic (unchanged when no resume_processor)
    if allow_search:
        tools = [TavilySearchTool(max_results=2)]
    else:
        tools = []

    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
    state = {"messages": query}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_message = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_message[-1]


# NEW: Easy wrapper class for backend
class ResumeAIAgent:
    def __init__(self, resume_path=RESUME_PDF_PATH):
        self.resume_path = resume_path
        self.resume_processor = None

        if os.path.exists(resume_path):
            try:
                print(f"📄 Loading resume from: {resume_path}")
                self.resume_processor = ResumeProcessor(resume_path)
                if self.resume_processor.setup_vector_store():
                    print("✅ Enhanced Resume AI Agent ready with improved RAG!")
                    print("🎯 Supports query types: skills, projects, experience, education, general")
                else:
                    print("❌ Failed to setup vector store")
            except Exception as e:
                print(f"❌ Error initializing agent: {e}")
        else:
            print(f"❌ Resume file not found: {resume_path}")

    def get_query_type(self, query):
        return get_query_type(query)

    def get_stats(self):
        """Get statistics about the loaded resume"""
        if not self.resume_processor or not self.resume_processor.vector_store:
            return {"status": "not_loaded"}

        # Get number of chunks in vector store
        try:
            vector_count = self.resume_processor.vector_store.index.ntotal
            return {
                "status": "loaded",
                "chunks_count": vector_count,
                "chunk_size": 1000,
                "chunk_overlap": 150,
                "retrieval_k": 7,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        except:
            return {"status": "loaded", "details": "unavailable"}

    def chat(self, query, llm_id="llama-3.3-70b-versatile", provider="Groq"):
        return get_response_from_ai_agent(
            llm_id=llm_id,
            query=query,
            allow_search=False,
            system_prompt="",
            provider=provider,
            resume_processor=self.resume_processor
        )

    def is_resume_question(self, query):
        return is_resume_related_query(query)



