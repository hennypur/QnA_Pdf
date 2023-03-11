import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfFileReader
import io


def load_pdf(file):
    pdf_reader = PdfReader(file)
    num_pages = pdf_reader.getNumPages()
    text = ""
    for page in range(num_pages):
        page_obj = pdf_reader.getPage(page)
        text += page_obj.extract_text()
    return text


def main():
    # Prompt user to upload a PDF file
    file = st.file_uploader("Upload a PDF file", type="pdf")

    # Generate summary if file is uploaded
    if file is not None:
        # Read PDF content
        text = load_pdf(file)

        # Define question answering pipeline
        choose_model = "distilbert-base-cased-distilled-squad"
        qa = pipeline("question-answering", model=choose_model)

        # Define a function to ask and answer questions
        def ask_question(question, text, qa):
            result = qa(question=question, context=text)
            answer = result["answer"]
            return answer

        # Define a list of example questions
        questions = [
            "What is the settlement agreement about?",
            "When was the Date of loss?",
            "Who are Our Client?",
            "What is Our Client's Gender?",
            "What is Our Client's Date of Birth?",
            "Who are the PERSON names in MEDICAL TREATMENTS?",
            "When was the earliest Date of Treatment?",
            "Did date of Treatments more than 30 days after Date of Loss?",
            "What are the most severe damage Our client's have?",
            "What are the mental, emotions, and other non-physical damage Our Client's have?"
        ]

        # Display example questions and answers
        st.write("Here are some example QnA of the PDF content:")
        for question in questions:
            answer = ask_question(question, text, qa)
            st.write(f"Q: {question}")
            st.write(f"A: {answer}\n")

        # Allow user to input custom question
        st.write("Enter your question below:")
        user_question = st.text_input("", "")
        if user_question:
            answer = ask_question(user_question, text, qa)
            st.write(f"Q: {user_question}")
            st.write(f"A: {answer}\n")


if __name__ == "__main__":
    main()
