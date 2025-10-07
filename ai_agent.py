import os

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


#   step1
GROQ_API_KEY = os.environ['GROQ_API_KEY']
TAVILY_API_KEY=os.environ['TAVILY_API_KEY']
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']


# NEW: Resume configuration
RESUME_PDF_PATH = "./MT24147_Yash choudhery.docx"  # ← CHANGE THIS TO YOUR ACTUAL RESUME PATH


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
            else:
                raise ValueError("Resume must be .pdf, .doc, or .docx file")

            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks")

            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

            print("Resume processing complete!")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def get_relevant_context(self, query):
        if not self.retriever:
            return ""
        try:
            docs = self.retriever.get_relevant_documents(query)
            return "\n\n".join([doc.page_content for doc in docs])
        except:
            return ""


def is_resume_related_query(query):
    keywords = ["experience", "work", "job", "skills", "education", "background",
                "projects", "resume", "cv", "programming", "about you"]
    return any(keyword in query.lower() for keyword in keywords)


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
        print("Using RAG...")
        context = resume_processor.get_relevant_context(query)

        if context:
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an AI representing a job candidate. 
                Answer based on their resume.

                Context: {context}"""),
                ("human", "{input}")
            ])

            document_chain = create_stuff_documents_chain(llm, rag_prompt)
            rag_chain = create_retrieval_chain(resume_processor.retriever, document_chain)
            response = rag_chain.invoke({"input": query})
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
                self.resume_processor = ResumeProcessor(resume_path)
                if self.resume_processor.setup_vector_store():
                    print("Resume AI Agent ready!")
            except Exception as e:
                print(f"Error: {e}")

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



