import os
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from Technical_Voice_Assistant.functions import transcribe_text_to_voice, chat_completion_call, text_to_speech_ai
from PIL import Image

api_key = os.getenv("OPENAI_API_KEY")



if api_key is None:
    raise ValueError(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

def main():

    favicon = Image.open("favicon.png")
    st.set_page_config(
        page_title="GenAI Demo | Trigent AXLR8 Labs",
        page_icon=favicon,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar Logo
    logo_html = """
    <style>
        [data-testid="stSidebarNav"] {
            background-image: url(https://trigent.com/wp-content/uploads/Trigent_Axlr8_Labs.png);
            background-repeat: no-repeat;
            background-position: 20px 20px;
            background-size: 80%;
        }
    </style>
    """
    st.sidebar.markdown(logo_html, unsafe_allow_html=True)

    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if "audio_counter" not in st.session_state:
        st.session_state["audio_counter"] = 0

    st.title("Technical voice assistant 💬")

    
    if api_key:
        success_message_html = """
        <span style='color:green; font-weight:bold;'>✅ Powering the Chatbot using Open AI's 
        <a href='https://platform.openai.com/docs/models/gpt-3-5' target='_blank'>gpt-3.5-turbo-0613 model</a>!</span>
        """

        # Display the success message with the link
        st.markdown(success_message_html, unsafe_allow_html=True)
        openai_api_key = api_key
    else:
        openai_api_key = st.text_input(
            'Enter your OPENAI_API_KEY: ', type='password')
        if not openai_api_key:
            st.warning('Please, enter your OPENAI_API_KEY', icon='⚠️')
        else:
            st.success('Ask Tech voice assistant about your software.', icon='👉')

    st.markdown("""
            ## **Bringing you to Ai Tech Support**
            
        **The Technical Assistant all you need 
        for your software related queries of our organization.
        Please feel free to ask any questions you have.**

------------------------------------------------------------------------------------------
    """)



    audio_bytes = audio_recorder(text="Record your issue here and please wait",
                                 recording_color="#e8b62c", neutral_color="#6aa36f", icon_size="2x")
    if audio_bytes:
        with st.spinner("Thinking.."):
            audio_location="audios/audio_file.wav"  #This is saif voice 
            with open(audio_location, "wb") as f:
                f.write(audio_bytes)

            text = transcribe_text_to_voice(audio_location)
            st.session_state['chat_history'].append({'role': 'user', 'content': text})

            api_response = chat_completion_call(text)
            st.session_state['chat_history'].append({'role': 'assistant', 'content': api_response})

            reversed_chat_history = st.session_state['chat_history'][::-1]
            for message in reversed_chat_history:
                with st.empty() and st.chat_message(message["role"]):
                    st.markdown(message['content'])

                    if message["role"] == "assistant":
                        audio_data = text_to_speech_ai(message['content'])
                        st.audio(audio_data, format='audio/mp3')


    
    # Footer
    footer_html = """
    <div style="text-align: right; margin-right: 10%;">
        <p>
            Copyright © 2024, Trigent Software, Inc. All rights reserved. | 
            <a href="https://www.facebook.com/TrigentSoftware/" target="_blank">Facebook</a> |
            <a href="https://www.linkedin.com/company/trigent-software/" target="_blank">LinkedIn</a> |
            <a href="https://www.twitter.com/trigentsoftware/" target="_blank">Twitter</a> |
            <a href="https://www.youtube.com/channel/UCNhAbLhnkeVvV6MBFUZ8hOw" target="_blank">YouTube</a>
        </p>
    </div>
    """

    # Custom CSS to make the footer sticky
    footer_css = """
    <style>
    .footer {
        position: fixed;
        z-index: 1000;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
    }
    [data-testid="stSidebarNavItems"] {
        max-height: 100%!important;
    }
    </style>
    """

    # Combining the HTML and CSS
    footer = f"{footer_css}<div class='footer'>{footer_html}</div>"

    # Rendering the footer
    st.markdown(footer, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
