from openai import OpenAI
import streamlit as st
import io
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from Technical_Voice_Assistant.prompt import template
from langchain_groq.chat_models import ChatGroq


msgs = StreamlitChatMessageHistory(key="special_app_key")

if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)
chain = prompt | ChatGroq(api_key=st.secrets.get("GROQ_API_KEY"))
# chain = prompt | ChatOpenAI(api_key=st.secrets.get('OPENAI_API_KEY'))
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,  # Always return the instance created earlier
    input_messages_key="question",
    history_messages_key="history",
)


def transcribe_text_to_voice(audio_location):
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
    audio_file = open(audio_location, "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript.text


def chat_completion_call(text):
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": text}, config)
    return response.content


def text_to_speech_ai(api_response):
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
    response = client.audio.speech.create(
        model="tts-1", voice="nova", input=api_response
    )

    if hasattr(response, "content") and response.content:
        audio_data = io.BytesIO(response.content)
        return audio_data
    else:
        return None
