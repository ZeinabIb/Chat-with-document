import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import fitz  # PyMuPDF for PDF handling
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("mrm8488/spanbert-finetuned-squadv1")
model = AutoModelForQuestionAnswering.from_pretrained(
    "mrm8488/spanbert-finetuned-squadv1"
)


def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text()

    return full_text


def answer_question(question, text):
    inputs = tokenizer.encode_plus(
        question, text, add_special_tokens=True, return_tensors="pt"
    )
    input_ids = inputs["input_ids"].tolist()[0]

    # Perform question answering
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    )
    return answer


def main():
    st.title("PDF Document Chatbot")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        full_text = extract_text_from_pdf(uploaded_file)

        st.write("### Uploaded PDF Content:")
        st.write(full_text)

        question = st.text_input("Ask a question about the document:")

        if st.button("Get Answer"):
            if question:
                answer = answer_question(question, full_text)
                st.write(f"### Answer:")
                st.write(answer)
            else:
                st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
