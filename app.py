import os
import re
import time

import streamlit as st
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)
from youtube_transcript_api.proxies import WebshareProxyConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
LLM_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
PREFERRED_LANGS = ["ar", "en"]


def get_secret(name, default=None):
    """Read a key from Streamlit secrets or environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


GROQ_API_KEY = get_secret("GROQ_API_KEY")
PROXY_USER = get_secret("WEBSHARE_PROXY_USERNAME")
PROXY_PASS = get_secret("WEBSHARE_PROXY_PASSWORD")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_video_id(url_or_id):
    """Accepts any YouTube link format (or a bare ID) and returns the 11-char ID."""
    text = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", text):
        return text
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"youtube\.com/(?:shorts|embed|live)/([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1)
    return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_transcript(video_id):
    """Returns (transcript_text, language_code). Raises on failure."""
    if PROXY_USER and PROXY_PASS:
        ytt_api = YouTubeTranscriptApi(
            proxy_config=WebshareProxyConfig(
                proxy_username=PROXY_USER,
                proxy_password=PROXY_PASS,
            )
        )
    else:
        ytt_api = YouTubeTranscriptApi()

    transcript_list = ytt_api.list(video_id)
    try:
        t = transcript_list.find_transcript(PREFERRED_LANGS)
    except NoTranscriptFound:
        t = next(iter(transcript_list))  # any available language
    fetched = t.fetch()
    text = " ".join(s.text for s in fetched)
    return text, t.language_code


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@st.cache_resource(show_spinner=False, max_entries=20)
def build_vectorstore(video_id, transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    return FAISS.from_documents(chunks, load_embeddings())


@st.cache_resource(show_spinner=False)
def load_llm():
    # max_retries: on a 429 (rate limit) the client waits and retries automatically
    return ChatGroq(model=LLM_MODEL, temperature=0.2, api_key=GROQ_API_KEY, max_retries=8)


def invoke_with_wait(chain, inputs, attempts=4):
    """Invoke a chain; if Groq's per-minute token limit is hit, wait and try again."""
    for i in range(attempts):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if "rate_limit" not in str(e).lower() or i == attempts - 1:
                raise
            time.sleep(20)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


QA_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant that answers questions about a YouTube video.
Use ONLY the transcript excerpts below. If they don't contain the answer, say you don't know.
Answer in the same language the user used in the question.

Transcript excerpts:
{context}

Recent conversation:
{history}

Question: {question}
Answer:"""
)

SUMMARY_PART_PROMPT = PromptTemplate.from_template(
    "Summarize this part of a video transcript in at most 5 short bullet points, "
    "in the same language as the transcript:\n\n{text}"
)

SUMMARY_FINAL_PROMPT = PromptTemplate.from_template(
    """Below are summaries of consecutive parts of one video.
Write one clear overall summary of the whole video: a short intro sentence, then the key points as bullets.
Write it in {language}.

{text}"""
)


def summarize(transcript, language_name):
    llm = load_llm()
    parser = StrOutputParser()
    # Small parts, one at a time, so each request stays under Groq's free-tier
    # tokens-per-minute limit (8000 TPM).
    splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=0)
    parts = splitter.split_text(transcript)

    if len(parts) == 1:
        partial = [transcript]
    else:
        part_chain = SUMMARY_PART_PROMPT | llm | parser
        partial = []
        progress = st.progress(0.0, text="بلخّص أجزاء الفيديو...")
        for i, p in enumerate(parts):
            partial.append(invoke_with_wait(part_chain, {"text": p}))
            progress.progress((i + 1) / len(parts), text=f"جزء {i + 1} من {len(parts)}")
        progress.empty()

    final_chain = SUMMARY_FINAL_PROMPT | llm | parser
    return invoke_with_wait(final_chain, {"text": "\n\n".join(partial), "language": language_name})


def answer_question(vectorstore, question, history_msgs):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    history = "\n".join(f"{m['role']}: {m['content']}" for m in history_msgs[-4:]) or "(none)"

    chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "history": RunnableLambda(lambda _: history),
            }
        )
        | QA_PROMPT
        | load_llm()
        | StrOutputParser()
    )
    return chain.stream(question)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="YouTube ChatBot", page_icon="🎬", layout="centered")
st.title("🎬 YouTube ChatBot")
st.caption("حط لينك أي فيديو على يوتيوب واسأل عن محتواه")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY مش متظبط. ضيفه في Secrets بتاعة الاستضافة.")
    st.stop()

if "video_id" not in st.session_state:
    st.session_state.video_id = None
    st.session_state.messages = []

with st.form("video_form"):
    url = st.text_input("لينك الفيديو", placeholder="https://www.youtube.com/watch?v=...")
    submitted = st.form_submit_button("تحميل الفيديو")

if submitted:
    vid = extract_video_id(url or "")
    if not vid:
        st.error("اللينك ده مش لينك يوتيوب صحيح.")
    else:
        with st.spinner("بجيب نص الفيديو..."):
            try:
                transcript, lang = fetch_transcript(vid)
            except TranscriptsDisabled:
                st.error("الفيديو ده مفيهوش ترجمة (captions)، فمش هقدر أقرا محتواه.")
                st.stop()
            except CouldNotRetrieveTranscript as e:
                name = type(e).__name__
                if name in ("RequestBlocked", "IpBlocked"):
                    st.error(
                        "يوتيوب رافض الطلبات من السيرفر ده. "
                        "ضيف بيانات Webshare proxy في الـ Secrets (شوف README)."
                    )
                else:
                    st.error(f"مقدرتش أجيب نص الفيديو ({name}).")
                st.stop()
            except StopIteration:
                st.error("الفيديو ده مفيهوش أي ترجمة.")
                st.stop()

        with st.spinner("بجهز الفيديو للأسئلة..."):
            build_vectorstore(vid, transcript)

        st.session_state.video_id = vid
        st.session_state.transcript = transcript
        st.session_state.lang = lang
        st.session_state.messages = []

if st.session_state.video_id:
    vid = st.session_state.video_id
    transcript = st.session_state.transcript
    vectorstore = build_vectorstore(vid, transcript)

    with st.expander("الفيديو", expanded=False):
        st.video(f"https://www.youtube.com/watch?v={vid}")
        st.caption(f"لغة الترجمة: {st.session_state.lang} · {len(transcript):,} حرف")

    if st.button("📝 لخّص الفيديو"):
        lang_name = "Arabic" if st.session_state.lang.startswith("ar") else "the transcript's language"
        st.session_state.messages.append({"role": "user", "content": "لخّص الفيديو"})
        try:
            with st.spinner("بلخّص الفيديو..."):
                summary = summarize(transcript, lang_name)
            st.session_state.messages.append({"role": "assistant", "content": summary})
        except Exception as e:
            st.session_state.messages.pop()
            if "rate_limit" in str(e).lower():
                st.error("وصلنا للحد المسموح من Groq في الدقيقة. استنى دقيقة وجرّب تاني.")
            else:
                st.error(f"حصل خطأ أثناء التلخيص: {type(e).__name__}")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    question = st.chat_input("اسأل أي حاجة عن الفيديو...")
    if question:
        history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                reply = st.write_stream(answer_question(vectorstore, question, history))
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.session_state.messages.pop()
                if "rate_limit" in str(e).lower():
                    st.error("وصلنا للحد المسموح من Groq في الدقيقة. استنى دقيقة وجرّب تاني.")
                else:
                    st.error(f"حصل خطأ: {type(e).__name__}")
else:
    st.info("ابدأ بلينك فيديو فوق 👆")
