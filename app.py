import streamlit as st

from src.transcript_loader import get_transcript
from src.splitter import split_transcript
from src.model import get_llm
from src.prompt import chat_prompt
from src.retriever import get_multi_query_retriever
from src.vectorstore import get_vector, add_chunks

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ================= GLOBAL ================= */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(99, 102, 241, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(14, 165, 233, 0.08),
                transparent 30%
            ),
            #0b0f19;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* ================= SIDEBAR ================= */

    section[data-testid="stSidebar"] {
        background: #090d16;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .brand {
        font-size: 27px;
        font-weight: 800;
        color: white;
        margin-bottom: 4px;
    }

    .brand-sub {
        font-size: 13px;
        color: #8c96aa;
        line-height: 1.5;
        margin-bottom: 22px;
    }

    .side-card {
        padding: 15px;
        border-radius: 14px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        margin-bottom: 10px;
    }

    .side-title {
        color: white;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 9px;
    }

    .side-text {
        color: #8d97ab;
        font-size: 12px;
        line-height: 1.6;
    }

    .pipeline-item {
        padding: 9px 10px;
        margin: 6px 0;
        border-radius: 9px;
        background: rgba(255,255,255,0.035);
        color: #c6cede;
        font-size: 12px;
    }

    /* ================= HERO ================= */

    .hero {
        padding: 28px 32px;
        border-radius: 22px;
        background:
            linear-gradient(
                135deg,
                rgba(99,102,241,0.16),
                rgba(14,165,233,0.08)
            );
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 22px;
    }

    .badge {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 999px;
        background: rgba(99,102,241,0.14);
        border: 1px solid rgba(99,102,241,0.25);
        color: #a5b4fc;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #ffffff;
        margin-bottom: 7px;
    }

    .hero-subtitle {
        color: #9ca7ba;
        font-size: 14px;
        line-height: 1.6;
    }

    /* ================= SECTION ================= */

    .section-title {
        font-size: 18px;
        font-weight: 750;
        color: white;
        margin-top: 8px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        font-size: 12px;
        color: #7f899d;
        margin-bottom: 13px;
    }

    /* ================= INPUT ================= */

    div[data-baseweb="input"] {
        border-radius: 12px;
    }

    div[data-baseweb="textarea"] {
        border-radius: 12px;
    }

    /* ================= BUTTON ================= */

    div.stButton > button {
        width: 100%;
        height: 46px;
        border-radius: 12px;
        font-weight: 700;
        border: 1px solid rgba(99,102,241,0.35);
        background: linear-gradient(
            135deg,
            #6366f1,
            #4f46e5
        );
        color: white;
    }

    div.stButton > button:hover {
        border-color: #818cf8;
        box-shadow: 0 10px 28px rgba(79,70,229,0.22);
    }

    /* ================= CHAT ================= */

    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        margin-bottom: 8px;
    }

    div[data-testid="stChatInput"] {
        padding-bottom: 10px;
    }

    /* ================= STATUS ================= */

    .video-ready {
        padding: 11px 14px;
        border-radius: 12px;
        background: rgba(16,185,129,0.08);
        border: 1px solid rgba(16,185,129,0.18);
        color: #9ae6c5;
        font-size: 12px;
        margin: 10px 0 18px 0;
    }

    .video-loading {
        padding: 11px 14px;
        border-radius: 12px;
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.18);
        color: #b8bdfc;
        font-size: 12px;
        margin: 10px 0 18px 0;
    }

    /* ================= FOOTER ================= */

    .footer {
        text-align: center;
        color: #566176;
        font-size: 11px;
        margin-top: 35px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False

if "video_url" not in st.session_state:
    st.session_state.video_url = ""

if "chain" not in st.session_state:
    st.session_state.chain = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# ============================================================
# CACHED RESOURCES
# ============================================================

@st.cache_resource
def load_vectorstore():
    return get_vector()


@st.cache_resource
def load_llm():
    return get_llm()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">🎬 YouTube RAG</div>

        <div class="brand-sub">
            Your AI chatbot for understanding
            YouTube videos with RAG.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="side-card">
            <div class="side-title">⚡ How it works</div>

            <div class="side-text">
                Paste a YouTube URL, process its transcript,
                and chat with the video using semantic retrieval.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🧠 RAG Pipeline")

    pipeline = [
        "📥 YouTube Transcript",
        "✂️ Recursive Text Splitting",
        "🧠 HuggingFace Embeddings",
        "🗄️ Chroma Vector Store",
        "🔎 MultiQueryRetriever",
        "🤖 LLM Generation",
    ]

    for item in pipeline:
        st.markdown(
            f'<div class="pipeline-item">{item}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown(
        """
        <div class="side-card">

            <div class="side-title">🛠️ Tech Stack</div>

            <div class="side-text">
                Python<br>
                LangChain<br>
                ChromaDB<br>
                HuggingFace<br>
                Streamlit<br>
                OpenAI-compatible LLM
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.video_loaded:

        st.markdown("### 📌 Current Video")

        st.caption(st.session_state.video_url)

        if st.button("🗑️ Clear Chat", use_container_width=True):

            st.session_state.messages = []

            st.rerun()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="badge">
            ⚡ AI • RAG • MULTI-QUERY
        </div>

        <div class="hero-title">
            Chat with Any YouTube Video
        </div>

        <div class="hero-subtitle">
            Turn a YouTube video into a knowledge base
            and ask questions about it like a chatbot.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# VIDEO INPUT
# ============================================================

st.markdown(
    '<div class="section-title">🎥 Add YouTube Video</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    'Paste the YouTube URL and process the transcript once.'
    '</div>',
    unsafe_allow_html=True,
)

url_col, button_col = st.columns([4, 1], gap="medium")

with url_col:

    youtube_url = st.text_input(
        "YouTube URL",
        value=st.session_state.video_url,
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )

with button_col:

    process_video = st.button(
        "🚀 Process Video",
        use_container_width=True,
    )


# ============================================================
# PROCESS VIDEO
# ============================================================

if process_video:

    if not youtube_url.strip():

        st.error("Please enter a YouTube URL.")

    else:

        try:

            vectorstore = load_vectorstore()

            transcript = RunnableLambda(get_transcript)

            splitter = RunnableLambda(split_transcript)

            add_to_vectorstore = RunnableLambda(
                lambda chunks: add_chunks(
                    vectorstore,
                    chunks,
                    chunks[0].metadata["video_id"],
                )
            )

            script_chain = (
                transcript
                | splitter
                | add_to_vectorstore
            )

            with st.status(
                "🔄 Processing video...",
                expanded=True,
            ) as status:

                st.write("📥 Fetching YouTube transcript...")

                st.write("✂️ Splitting transcript...")

                st.write("🧠 Creating embeddings...")

                st.write("🗄️ Updating Chroma vector store...")

                script_chain.invoke(youtube_url)

                st.write("🔎 Creating MultiQueryRetriever...")

                retriever = get_multi_query_retriever(
                    vectorstore
                )

                # -----------------------------
                # Parallel Retrieval
                # -----------------------------

                parallel_chain = RunnableParallel(
                    {
                        "context": retriever,
                        "question": RunnablePassthrough(),
                    }
                )

                # -----------------------------
                # RAG Chain
                # -----------------------------

                prompt = chat_prompt()

                llm = load_llm()

                parser = StrOutputParser()

                chain = (
                    parallel_chain
                    | prompt
                    | llm
                    | parser
                )

                # Save everything in session
                st.session_state.vectorstore = vectorstore
                st.session_state.chain = chain
                st.session_state.video_loaded = True
                st.session_state.video_url = youtube_url

                # New video = fresh chat
                st.session_state.messages = []

                status.update(
                    label="✅ Video is ready!",
                    state="complete",
                    expanded=False,
                )

            st.success(
                "Video processed successfully. "
                "You can now start chatting."
            )

            st.rerun()

        except Exception as e:

            st.error(
                "❌ Failed to process the YouTube video."
            )

            with st.expander("🔧 Technical Details"):
                st.exception(e)


# ============================================================
# VIDEO PREVIEW
# ============================================================

if st.session_state.video_loaded:

    st.markdown(
        """
        <div class="video-ready">
            🟢 Video processed successfully — RAG chatbot is ready.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🎥 Show YouTube Video", expanded=False):

        st.video(st.session_state.video_url)


# ============================================================
# CHAT HEADER
# ============================================================

st.markdown(
    '<div class="section-title">💬 Video Chat</div>',
    unsafe_allow_html=True,
)

if st.session_state.video_loaded:

    st.markdown(
        '<div class="section-subtitle">'
        'Ask multiple questions about the processed video.'
        '</div>',
        unsafe_allow_html=True,
    )

else:

    st.info(
        "👆 First paste a YouTube URL and click "
        "**Process Video** to start chatting."
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

if st.session_state.video_loaded:

    user_query = st.chat_input(
        "Ask anything about the video..."
    )

    if user_query:

        # -----------------------------
        # User Message
        # -----------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_query,
            }
        )

        with st.chat_message("user"):

            st.markdown(user_query)

        # -----------------------------
        # Assistant Message
        # -----------------------------

        with st.chat_message("assistant"):

            with st.spinner("🔎 Searching the video..."):

                try:

                    answer = st.session_state.chain.invoke(
                        user_query
                    )

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                except Exception as e:

                    st.error(
                        "❌ Failed to generate the answer."
                    )

                    with st.expander(
                        "🔧 Technical Details"
                    ):

                        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Built with Python • LangChain • ChromaDB • Streamlit • RAG
    </div>
    """,
    unsafe_allow_html=True,
)