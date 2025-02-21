import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from Technical_Voice_Assistant.functions import (
    transcribe_text_to_voice,
    chat_completion_call,
    text_to_speech_ai,
)
from PIL import Image
from dotenv import load_dotenv

load_dotenv(override=True)

def main():
    # Set page configuration
    favicon = Image.open("favicon.png")
    st.set_page_config(
        page_title="Tech Voice Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Detailed sidebar content with descriptive information and contact details
    sidebar_content = """
    # Tech Voice Assistant

    ## About
    Tech Voice Assistant is your personal AI-powered technical support tool.  
    It leverages advanced voice recognition and AI capabilities to provide quick and accurate answers to your software-related queries.  
    Simply record your issue and let the assistant guide you with troubleshooting and solutions.

    ## Usage Instructions
    1. Click the record button below.
    2. Clearly state your software-related issue.
    3. Wait as the assistant processes your query.
    4. Receive both a text and audio response with helpful information.

    ## Contact
    For inquiries, feedback, or support, please connect with us:
    - **GitHub:** [alsaif1431](https://github.com/alsaif1431)
    - **LinkedIn:** [Saif Pasha](https://www.linkedin.com/in/saif-pasha-59643b197/)
    """
    st.sidebar.markdown(sidebar_content, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "audio_counter" not in st.session_state:
        st.session_state["audio_counter"] = 0

    st.title("Technical Voice Assistant 💬")


    st.markdown(
        """
        ## Bringing you to AI Tech Support

        Your Technical Assistant for all your software-related queries.  
        Please feel free to ask any questions you have.
        
        ------------------------------------------------------------------------------------------
        """
    )

    audio_bytes = audio_recorder(
        text="Record your issue here and please wait",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_size="2x",
    )
    if audio_bytes:
        with st.spinner("Thinking..."):
            audio_location = "audios/audio_file.wav"  # Recorded voice file location
            with open(audio_location, "wb") as f:
                f.write(audio_bytes)

            text = transcribe_text_to_voice(audio_location)
            st.session_state["chat_history"].append({"role": "user", "content": text})

            api_response = chat_completion_call(text)
            st.session_state["chat_history"].append({"role": "assistant", "content": api_response})

            reversed_chat_history = st.session_state["chat_history"][::-1]
            for message in reversed_chat_history:
                with st.empty() and st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant":
                        audio_data = text_to_speech_ai(message["content"])
                        st.audio(audio_data, format="audio/mp3")




if __name__ == "__main__":
    main()
