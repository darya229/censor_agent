import streamlit as st
import docx
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from markdown_pdf import MarkdownPdf, Section
from io import BytesIO

def generate_pdf(markdown_content):
    pdf = MarkdownPdf()
    pdf.meta["title"] = 'Отчет'
    pdf.meta["author"] = 'AI Assistant'
    pdf.add_section(Section(markdown_content, toc=False))
    return pdf

deepseek_llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key="...",
    temperature=1,
    streaming=True
)

st.subheader('ИИ-помощник: цензор')

if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "text": "Заргузите документ, который необходимо проверить"}]

for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message['role']):
        # Отображаем текстовый контент
        if 'text' in message:
            st.write(message['text'])

# React to user input
user_input = st.chat_input('Введите дополнительные инструкции или оставьте поле пустым', accept_file=True, accept_audio=False)
if user_input:
    doc = docx.Document(user_input.files[0])
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    content = '\n'.join(full_text)
    # Display user message
    with st.chat_message('user'):
        st.write(content)
        # st.markdown(prompt)

    response = f'Echo: Обработка запроса'

    with st.chat_message('assistant'):
        messages = [
        SystemMessage(content="Ты полезный ассистент."),
        HumanMessage(content="Расскажи коротко как работает RAG-система")]
        def generate_response():
            for chunk in deepseek_llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        response = st.write_stream(generate_response)
        download_content = generate_pdf(response)

        # Сохраняем в буфер
        buffer = BytesIO()
        download_content.save(buffer)
        buffer.seek(0)
        st.download_button(
            label="📥 Скачать PDF",
            data=buffer.getvalue(),
            file_name="отчет.pdf",
            mime="application/pdf",
            key="download_pdf",
            on_click="ignore"
        )


