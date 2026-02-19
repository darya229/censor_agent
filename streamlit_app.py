import streamlit as st
import docx
from langchain_core.messages import SystemMessage, HumanMessage
from prompts import SYSTEM_PROMPT_v1, SYSTEM_PROMPT_v2, USER_PROMT, RULES, PROMPT_SENTIMENT
from langchain_deepseek import ChatDeepSeek
from markdown_pdf import MarkdownPdf, Section
from io import BytesIO
from dotenv import load_dotenv
from loguru import logger 
import plotly.io as pio
import base64
from io import BytesIO

load_dotenv()
import os
import json
import plotly.express as px
import pandas as pd
from datetime import datetime

import time
pio.kaleido.scope.mathjax = None  # Отключаем MathJax если не нужен
def generate_pdf(markdown_content, filename, plotly_fig=None, df = None):
    pdf = MarkdownPdf()
    pdf.meta["title"] = 'Отчет'
    pdf.meta["author"] = 'AI Assistant'
    pdf_content = f"Отчет: {filename}\n\n "
    pdf_content += "*Анализ сентимента*"


    if plotly_fig:
        # Конвертируем plotly график в base64 изображение
        img_bytes = pio.to_image(plotly_fig, format='png', engine='orca')
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        pdf_content += f"![График сентимента](data:image/png;base64,{img_base64})\n\n"

    if df is not None and not df.empty:
        pdf_content += df.to_markdown()

    pdf_content += "#Анализ аналитического отчета  #"

    # Добавляем текстовый контент
    pdf_content += markdown_content

    pdf.add_section(Section(pdf_content, toc=False))
    return pdf

deepseek_llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=st.secrets["MY_LLM"],
    temperature=1,
    streaming=True
)

deepseek_llm_not_streaming = ChatDeepSeek(
    model="deepseek-chat",
    api_key=st.secrets["MY_LLM"],
    temperature=1
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

            ### Готовим сентимент #####

            messages_sentiment = [HumanMessage(content=PROMPT_SENTIMENT.format(report_text = content))]
            response_sentiment = deepseek_llm.invoke(messages_sentiment)
            temp_message.empty()
            st.write("✅ Готов расчет сентимента")
            try:
                companies_sentiment = json.loads(response_sentiment.content)
            except:
                st.warning('Не удалось прочитать JSON объект')
                st.write(response_sentiment.content)

            df = pd.DataFrame(companies_sentiment)

            # Создаем точечную диаграмму
            fig = px.scatter(df, 
                            x='company', 
                            y='sentiment',
                            title='Сентимент по компаниям',
                            labels={'company': 'Компания', 'sentiment': 'Сентимент'},
                            size=[20] * len(df),  # Размер точек
                            color_discrete_sequence=['darkgrey'])

            # Добавляем горизонтальную линию на отметке 5
            fig.add_hline(y=5, 
                        line_dash="solid", 
                        line_color="black",
                        line_width=1,
                        annotation_text="нейтральный сентимент",
                        annotation_position="top left")

            # Настраиваем отображение
            fig.update_layout(
                xaxis_title="Компания",
                yaxis_title="Сентимент",
                showlegend=False,
                yaxis=dict(
                    range=[0, 10]  # Устанавливаем диапазон для лучшей видимости
                )
            )

            # Настраиваем внешний вид точек
            fig.update_traces(
                marker=dict(
                    size=15,  # Размер точек
                    line=dict(width=1, color='darkgrey')  # Обводка точек
                )
            )

            # Показываем график
            st.plotly_chart(fig)


            ### Анализируем текст ####
            temp_message = st.empty()
            temp_message.write("⏳ Анализ отчета...")
            system_instructions = SYSTEM_PROMPT_v1.format(rules=RULES, date = datetime.now().strftime("%Y-%m-%d"))
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
            download_content = generate_pdf(response, user_input.files[0].name, fig, df)

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




