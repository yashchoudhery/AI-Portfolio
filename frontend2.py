import streamlit as st
import requests
import os
# Replace hardcoded localhost with:
API_URL = os.getenv("API_URL", "https://AiPorfolio.onrender.com/chat")

# Page configuration — MUST be the very first Streamlit call
st.set_page_config(
    page_title="Yash Choudhery - Portfolio",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize session state for theme and query
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'query_input' not in st.session_state:
    st.session_state.query_input = ''

# Custom CSS for beautiful styling
def apply_custom_css():
    if st.session_state.theme == 'light':
        bg_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        card_bg = "#ffffff"
        text_color = "#2d3748"
        secondary_text = "#4a5568"
        input_bg = "#f7fafc"
        input_text_color = "#2d3748"
        placeholder_color = "#a0aec0"
        button_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
        shadow = "0 10px 30px rgba(0,0,0,0.1)"
        title_color = "#2d3748"  # Changed to dark color for light theme
        card_hover_bg = "#f7fafc"
        response_header_color = "#2d3748"  # For response header
        cursor_color = "#2d3748"  # For text cursor
    else:
        bg_gradient = "linear-gradient(135deg, #1a202c 0%, #2d3748 100%)"
        card_bg = "#2d3748"
        text_color = "#e2e8f0"
        secondary_text = "#cbd5e0"
        input_bg = "#1a202c"
        input_text_color = "#e2e8f0"
        placeholder_color = "#718096"
        button_gradient = "linear-gradient(135deg, #4299e1 0%, #667eea 100%)"
        shadow = "0 10px 30px rgba(0,0,0,0.3)"
        title_color = "#e2e8f0"  # Light color for dark theme
        card_hover_bg = "#4a5568"
        response_header_color = "#e2e8f0"  # For response header
        cursor_color = "#e2e8f0"  # For text cursor

    css = f"""
    <style>
        /* Main background */
        .stApp {{
            background: {bg_gradient};
        }}

        /* Remove top padding and white box */
        .block-container {{
            padding-top: 0rem !important;
            padding-bottom: 2rem !important;
        }}

        /* Container styling */
        .main-container {{
            background: {card_bg};
            border-radius: 20px;
            padding: 2.5rem;
            box-shadow: {shadow};
            margin: 0 auto;
            margin-top: 0 !important;
            max-width: 900px;
        }}

        /* Header styling */
        .portfolio-header {{
            text-align: center;
            margin-bottom: 2rem;
            padding-top: 0;
        }}

        .portfolio-title {{
            font-size: 3.5rem;
            font-weight: 800;
            color: {title_color} !important;
            margin-bottom: 0.5rem;
            margin-top: 0;
            padding-top: 0;
            animation: fadeInDown 1s ease;
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
        }}

        .portfolio-subtitle {{
            font-size: 1.3rem;
            color: {secondary_text} !important;
            font-weight: 300;
            animation: fadeInUp 1s ease;
        }}

        /* Fix for dynamic header text */
        .dynamic-header h1 {{
            color: {title_color} !important;
            font-size: 3.5rem !important;
            font-weight: 800 !important;
            margin-bottom: 0.5rem !important;
            margin-top: 0 !important;
            text-align: center !important;
            animation: fadeInDown 1s ease !important;
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3) !important;
        }}

        .dynamic-header p {{
            color: {title_color} !important;
            font-size: 1.3rem !important;
            font-weight: 300 !important;
            text-align: center !important;
            animation: fadeInUp 1s ease !important;
            margin-bottom: 2rem !important;
        }}

        /* Query section */
        .query-section {{
            margin: 2rem 0;
        }}

        .section-label {{
            font-size: 1.2rem;
            font-weight: 600;
            color: {text_color};
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
        }}

        .emoji-icon {{
            font-size: 1.5rem;
            margin-right: 0.5rem;
        }}

        /* Text area styling with visible text color and cursor */
        .stTextArea textarea {{
            background: {input_bg} !important;
            border: 2px solid #e2e8f0 !important;
            border-radius: 15px !important;
            font-size: 1rem !important;
            padding: 1rem !important;
            transition: all 0.3s ease !important;
            color: {input_text_color} !important;
            caret-color: {cursor_color} !important;
        }}

        /* Placeholder color styling */
        .stTextArea textarea::placeholder {{
            color: {placeholder_color} !important;
            opacity: 1 !important;
        }}

        .stTextArea textarea::-webkit-input-placeholder {{
            color: {placeholder_color} !important;
            opacity: 1 !important;
        }}

        .stTextArea textarea::-moz-placeholder {{
            color: {placeholder_color} !important;
            opacity: 1 !important;
        }}

        .stTextArea textarea:-ms-input-placeholder {{
            color: {placeholder_color} !important;
            opacity: 1 !important;
        }}

        .stTextArea textarea:focus {{
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
            caret-color: {cursor_color} !important;
        }}

        /* Button styling */
        .stButton button {{
            background: {button_gradient} !important;
            color: white !important;
            border: none !important;
            border-radius: 25px !important;
            padding: 0.75rem 2.5rem !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
            width: 100% !important;
        }}

        .stButton button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        }}

        /* Feature card buttons */
        .feature-card button {{
            background: transparent !important;
            color: {text_color} !important;
            border: 2px solid #667eea !important;
            border-radius: 15px !important;
            padding: 1.5rem !important;
            transition: all 0.3s ease !important;
            box-shadow: none !important;
            width: 100% !important;
            height: 120px !important;
        }}

        .feature-card button:hover {{
            background: {card_hover_bg} !important;
            transform: translateY(-5px) !important;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3) !important;
        }}

        /* Response section */
        .response-box {{
            background: {input_bg};
            border-radius: 15px;
            padding: 1.5rem;
            margin-top: 2rem;
            border-left: 4px solid #667eea;
            animation: slideIn 0.5s ease;
            color: {text_color};
        }}

        /* Response header styling */
        .response-header {{
            color: {response_header_color} !important;
            font-size: 1.5rem !important;
            font-weight: 600 !important;
            margin-bottom: 1rem !important;
        }}

        /* Fix markdown headers */
        .stMarkdown h3 {{
            color: {response_header_color} !important;
        }}

        .stMarkdown h1, .stMarkdown h2, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
            color: {text_color} !important;
        }}

        /* Theme toggle button */
        .theme-toggle {{
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
        }}

        .theme-toggle button {{
            background: {card_bg} !important;
            color: {text_color} !important;
            border: 2px solid #667eea !important;
            border-radius: 50% !important;
            width: 50px !important;
            height: 50px !important;
            font-size: 1.5rem !important;
            box-shadow: {shadow} !important;
            transition: all 0.3s ease !important;
        }}

        .theme-toggle button:hover {{
            transform: rotate(180deg) !important;
            background: {button_gradient} !important;
            color: white !important;
        }}

        /* Animations */
        @keyframes fadeInDown {{
            from {{
                opacity: 0;
                transform: translateY(-20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateX(-20px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}

        /* Error message styling */
        .stAlert {{
            border-radius: 10px !important;
            border-left: 4px solid #f56565 !important;
        }}

        /* Hide default Streamlit elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Apply custom CSS
apply_custom_css()

# Theme toggle button in top right corner
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("🌙" if st.session_state.theme == 'light' else "☀️", key="theme_toggle", help="Toggle Dark/Light Mode"):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
        st.rerun()

# Header section with dynamic colors
st.markdown("""
    <div class="dynamic-header">
        <h1>Hello, I'm</h1>
        <h1>✨Yash Choudhery</h1>
        <p>Ask me anything about my skills, qualifications & experience!</p>
    </div>
""", unsafe_allow_html=True)

# Divider with gradient
st.markdown("""
    <div style="height: 3px; background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%); 
    border-radius: 10px; margin: 2rem 0;"></div>
""", unsafe_allow_html=True)

# Feature cards - Clickable to fill query
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    if st.button("🎓\n\n**Qualifications**", key="qual_btn", help="Click to ask about qualifications"):
        st.session_state['query_input'] = "What are Yash Choudhery's educational qualifications and certifications?"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    if st.button("💻\n\n**Technical Skills**", key="skills_btn", help="Click to ask about technical skills"):
        st.session_state['query_input'] = "What are Yash Choudhery's technical skills and areas of expertise?"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    if st.button("🚀\n\n**Projects**", key="proj_btn", help="Click to ask about projects"):
        st.session_state['query_input'] = "What projects has Yash Choudhery worked on and what were his contributions?"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Query section
st.markdown("""
    <div class="section-label">
        <span class="emoji-icon">💬</span>
        <span>Your Query</span>
    </div>
""", unsafe_allow_html=True)

# Fixed text area - removed default value to avoid session state warning
query = st.text_area(
    "Your Query",
    height=150,
    value=st.session_state.get('query_input',''),  # Use session state value directly
    placeholder="💡 Try asking: What are Yash's technical skills? What projects has he worked on? What is his educational background?",
    label_visibility="collapsed",
    key="query_input"  # Changed key name to avoid conflicts
)

# Update session state when user types
if query != st.session_state.query_input:
    st.session_state.query_input = query

# Search button
if st.button("🔍 Search", key="search_btn"):
    if query.strip():
        with st.spinner("✨ Processing your query..."):
            API_URL = "http://127.0.0.1:9999/chat"
            payload = {
                "model_name": "llama-3.3-70b-versatile",
                "model_provider": "Groq",
                "prompt": "Act as AI Assistant",
                "messages": [query],
                "allow_search": False
            }

            try:
                response = requests.post(API_URL, json=payload)
                response_data = response.json()

                if response.status_code == 200:
                    # SUCCESS - Extract just the response text
                    ai_response = response_data.get('response', '')
                    is_resume = response_data.get('is_resume_related', False)

                    # Custom styled response header
                    st.markdown('<div class="response-header">🎯 Response</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="response-box">{ai_response}</div>', unsafe_allow_html=True)

                    # Optional: Show if resume-related (for debugging)
                    # st.caption(f"Resume-related: {is_resume}")
                else:
                    # ERROR
                    error_msg = response_data.get('detail', response_data.get('error', 'Unknown error'))
                    st.error(f"❌ {error_msg}")

            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")
    else:
        st.warning("⚠️ Please enter a query to search!")