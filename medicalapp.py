import PyPDF2
import docx
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
import wikipedia
import requests
from bs4 import BeautifulSoup
import json
import openai
import streamlit as st

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extractText()
    return text

def extract_text_from_word(doc_path):
    doc = docx.Document(doc_path)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text
    return text

def extract_text_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    text = ''
    for column in df.columns:
        text += ' '.join(df[column].astype(str).tolist())
    return text

def extract_text_from_youtube(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = ' '.join([item['text'] for item in transcript])
    return text

def extract_text_from_wikipedia(page_title):
    return wikipedia.page(page_title).content

def extract_text_from_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text(separator=' ')
    return text

def merge_texts(*texts):
    return ' '.join(texts)

# Extract texts from multiple sources
pdf_text = extract_text_from_pdf('path/to/pdf')
word_text = extract_text_from_word('path/to/word.docx')
excel_text = extract_text_from_excel('path/to/excel.xlsx')
youtube_text = extract_text_from_youtube('youtube_video_id')
wikipedia_text = extract_text_from_wikipedia('Wikipedia_Page_Title')
website_text = extract_text_from_website('https://hospitalwebsite.com')

# Merge all texts
merged_text = merge_texts(pdf_text, word_text, excel_text, youtube_text, wikipedia_text, website_text)

# Prepare training data
training_data = [
    {"prompt": "What are the symptoms of diabetes?", "completion": "The symptoms of diabetes include increased thirst, frequent urination, hunger, fatigue, and blurred vision."},
    {"prompt": "Tell me about diabetes.", "completion": merged_text[:1000]}  # Sample data
    # Add more data here
]

# Save training data to JSONL file
with open('training_data.jsonl', 'w') as f:
    for item in training_data:
        f.write(json.dumps(item) + '\n')

# Fine-tune the model
openai.api_key = 'YOUR_OPENAI_API_KEY'

response = openai.FineTune.create(
    training_file="training_data.jsonl",
    model="gpt-3.5-turbo",
    n_epochs=4
)
print(response)

# Streamlit app
st.title("Medical Chatbot")

user_query = st.text_input("Ask a medical question:")

if st.button("Get Answer"):
    if user_query:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",
            prompt=user_query,
            max_tokens=150
        )
        answer = response.choices[0].text.strip()
        st.write(answer)
    else:
        st.write("Please enter a question.")

if __name__ == "__main__":
    st.run()
