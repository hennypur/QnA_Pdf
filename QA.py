
import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader

def load_pdf(file):
    pdf_reader = PdfReader(file)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page in range(num_pages):
        page_obj = pdf_reader.pages[page]
        text += page_obj.extract_text()
    return text

def main():
    # Prompt user to upload a PDF file
    file = st.file_uploader("Upload a PDF file", type="pdf")

    # Generate summary if file is uploaded
    if file is not None:
        # Read PDF content
        with st.spinner('Extracting text from PDF...'):
            text = load_pdf(file)

        # Define question answering pipeline
        choose_model = "distilbert-base-cased-distilled-squad"
        with st.spinner('Loading model...'):
            qa = pipeline("question-answering", model=choose_model)

        # Define a function to ask and answer questions
        def ask_question(question, text, qa):
            result = qa(question=question, context=text)
            answer = result["answer"]
            return answer

        # Define a list of example questions
        questions = [
            "What are the most severe damage Our client's have?"
        ]

        # Display example questions and answers
        st.write("Here are some example QnA of the PDF content:")
        with st.spinner('Answering example questions...'):
            for question in questions:
                answer = ask_question(question, text, qa)
                st.write(f"Q: {question}")
                st.write(f"A: {answer}\n")

        # Allow user to input custom question
        st.write("Enter your question below:")
        user_question = st.text_input("", "")
        if user_question:
            with st.spinner('Answering your question...'):
                answer = ask_question(user_question, text, qa)
                st.write(f"Q: {user_question}")
                st.write(f"A: {answer}\n")


if __name__ == "__main__":
    st.set_page_config(page_title='PDF QnA', page_icon=':books:')
    st.title('PDF QnA')
    st.write('This app allows you to ask questions about a PDF document.')
    main()
