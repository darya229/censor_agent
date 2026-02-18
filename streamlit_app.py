import streamlit as st
import docx
from langchain_core.messages import SystemMessage, HumanMessage
from prompts import SYSTEM_PROMPT, USER_PROMT, RULES
from langchain_deepseek import ChatDeepSeek
from markdown_pdf import MarkdownPdf, Section
from io import BytesIO
from dotenv import load_dotenv

import os
from datetime import datetime

import time

def generate_pdf(markdown_content, filename):
    pdf = MarkdownPdf()
    pdf.meta["title"] = 'Отчет'
    pdf.meta["author"] = 'AI Assistant'
    pdf.add_section(Section(f"Отчет: {filename} \n\n {markdown_content}", toc=False))
    return pdf

deepseek_llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=st.secrets["MY_LLM"],
    temperature=1,
    streaming=True
)

# deepseek_llm = ChatDeepSeek(
#     model="deepseek-reasoner",
#     api_key=API_DEEPSEEK,
#     temperature=1,
#     max_tokens=32000,
#     reasoning_effort="medium",
#     streaming=True
# )

st.subheader('ИИ-помощник «Цензор»')

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "text": "Заргузите документ, который необходимо проверить"}]

for i, message in enumerate(st.session_state.messages):
    with st.chat_message("assistant", avatar=":material/priority_high:"):
        # Отображаем текстовый контент
        if 'text' in message:
            st.write(message['text'])

# React to user input
user_input = st.chat_input('Введите дополнительные инструкции или оставьте поле пустым', accept_file=True, accept_audio=False)
if user_input:
    if user_input.files:
        doc = docx.Document(user_input.files[0])
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        content = '\n'.join(full_text)
        # Display user message
        with st.chat_message("user", avatar=":material/person_pin:"):
            # st.write(user_input)
            st.write(f"{user_input.get("text", " ")}\n\n Прикрепленный файл: {user_input.files[0].name}")
            # st.markdown(prompt)

        with st.chat_message("ai", avatar=":material/android:"):
            temp_message = st.empty()
            temp_message.write("⏳ Обработка запроса...")
            system_instructions = SYSTEM_PROMPT.format(rules=RULES, date = datetime.now().strftime("%Y-%m-%d"))
            user_instructions = USER_PROMT.format(additional_instructions = user_input.get("text", " "),
                                                   analytical_report = content)
            # logger.info(f"Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} \n\n SYSTEM_PROMPT: \n\n {system_instructions} \n")
            messages = [
            SystemMessage(content=system_instructions),
            HumanMessage(content=user_instructions)]
            
            # st.write(messages)
            

            def generate_response():
                for chunk in deepseek_llm.stream(messages):
                    if chunk.content:
                        yield chunk.content
            if generate_response:
                temp_message.empty()
                st.write("✅ Ответ готов")

            response = st.write_stream(generate_response)
            download_content = generate_pdf(response, user_input.files[0].name)

            # Сохраняем в буфер
            buffer = BytesIO()
            download_content.save(buffer)
            buffer.seek(0)
            st.download_button(
                label="Скачать PDF",
                data=buffer.getvalue(),
                file_name=f"{os.path.splitext(user_input.files[0].name)[0][:25]}_результаты_проверки.pdf",
                mime="application/pdf",
                key="download_pdf",
                on_click="ignore",
                icon = ":material/download:"
            )
    else:
        st.warning("Пожалуйста, загрузите документ")



