from openai import OpenAI
import streamlit as st
import io
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

api_key = st.secrets["OPENAI_API_KEY"]

if api_key is None:
    raise ValueError(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable."
    )


template = """
    You are a Technical support agent for my software company.

    A customer calls in and complains that one of the most important 
    features or there is some other issue in the software which
    is breaking my software. 

    You checked and found out
    that it’s an issue on the customer’s end. 
    Now you will assist 
    them in troubleshooting the problem while maintaining empathy and patience.

    Remember that your sole purpose it to provide Technical support to the 
    customers.

    Do not generate extra conversations which may lead to unsatisfaction of 
    the customer.

    When asked for your name, you must respond with "Technical support Agent"
    and don't include the source nor say "thanks for reaching us" instead just
    tell him your name.

    Always say "Thanks for reaching us" at the end of the answer.

    Please do your best.
"""

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
chain = prompt | ChatOpenAI()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,  # Always return the instance created earlier
    input_messages_key="question",
    history_messages_key="history",
)


def transcribe_text_to_voice(audio_location):
    client = OpenAI()
    audio_file = open(audio_location, "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return transcript.text


def chat_completion_call(text):
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": text}, config)
    return response.content


def text_to_speech_ai(api_response):
    client = OpenAI()
    response = client.audio.speech.create(
        model="tts-1", voice="nova", input=api_response
    )

    if hasattr(response, "content") and response.content:
        audio_data = io.BytesIO(response.content)
        return audio_data
    else:
        return None
