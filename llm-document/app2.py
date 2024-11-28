import streamlit as st
import pdfplumber
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize


# PDF'den metin çıkarma fonksiyonu
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        extracted_text = ""
        for page in pdf.pages:
            extracted_text += page.extract_text()
    return extracted_text


# Özetleme fonksiyonu
def summarize_text(document_text, max_length, min_length=30):
    summarizer = pipeline("summarization", model="t5-small")
    summary = summarizer(document_text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']


# Soru üretimi pipeline'ı tanımlanıyor
qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")


# Soru üretim fonksiyonu
def generate_questions_pipeline(passage, min_questions=3):
    input_text = f"generate questions: {passage}"
    results = qg_pipeline(input_text)
    questions = results[0]['generated_text'].split('<sep>')

    # ensure we have at least 3 questions
    questions = [q.strip() for q in questions if q.strip()]

    # if fewer than 3 questions, try to regenerate from smaller parts of the passage
    if len(questions) < min_questions:
        passage_sentences = passage.split('. ')
        for i in range(len(passage_sentences)):
            if len(questions) >= min_questions:
                break
            additional_input = ' '.join(passage_sentences[i:i + 2])
            additional_results = qg_pipeline(f"generate questions: {additional_input}")
            additional_questions = additional_results[0]['generated_text'].split('<sep>')
            questions.extend([q.strip() for q in additional_questions if q.strip()])

    return questions[:min_questions]  # return only the top 3 questions


# Sayfa yapılandırması
st.sidebar.title("Navigasyon")
page = st.sidebar.radio("Sayfa Seçin", ["PDF Belge Yükleme", "PDF Belge Analizi", "PDF Belge Soru Üretimi",
                                        "PDF Belge Soru Cevaplama"])

# PDF Belge Yükleme Sayfası
if page == "PDF Belge Yükleme":
    st.title("PDF Belge Yükleme")
    uploaded_pdf = st.file_uploader("PDF dosyasını yükleyin", type="pdf")

    if uploaded_pdf is not None:
        # PDF'den metin çıkar
        extracted_text = extract_text_from_pdf(uploaded_pdf)

        # Metni göster
        st.subheader("PDF'den Çıkarılan Metin:")
        st.text_area("Metin", extracted_text, height=400)

        # Çıkarılan metni bir dosyaya kaydetme seçeneği
        st.download_button(
            label="Metni TXT Dosyası Olarak İndir",
            data=extracted_text,
            file_name="extracted_text.txt",
            mime="text/plain"
        )

# PDF Belge Analizi Sayfası
elif page == "PDF Belge Analizi":
    st.title("PDF Belge Analizi")

    uploaded_pdf = st.file_uploader("PDF dosyasını yükleyin", type="pdf")

    if uploaded_pdf is not None:
        # PDF'den metin çıkar
        extracted_text = extract_text_from_pdf(uploaded_pdf)

        # Kullanıcıdan özetin uzunluğunu ayarlamak için bir slider
        max_length = 150

        # Metnin özetlenmesi
        if len(extracted_text) > 0:
            summary = summarize_text(extracted_text[:1000], max_length=max_length)  # ilk 1000 karakteri özetle
            st.subheader("Özet:")
            st.write(summary)
        else:
            st.warning("Lütfen bir PDF dosyası yükleyin ve özetleme için bekleyin.")

# PDF Belge Soru Üretimi Sayfası
elif page == "PDF Belge Soru Üretimi":
    st.title("PDF Belge Soru Üretimi")

    uploaded_pdf = st.file_uploader("PDF dosyasını yükleyin", type="pdf")

    if uploaded_pdf is not None:
        # PDF'den metin çıkar
        extracted_text = extract_text_from_pdf(uploaded_pdf)

        # Split the text into sentences
        sentences = sent_tokenize(extracted_text)

        # Combine sentences into passages
        passages = []
        current_passage = ""
        for sentence in sentences:
            if len(current_passage.split()) + len(sentence.split()) < 200:  # adjust the word limit as needed
                current_passage += " " + sentence
            else:
                passages.append(current_passage.strip())
                current_passage = sentence
        if current_passage:
            passages.append(current_passage.strip())

        # Slider to allow the user to select the number of passages to display
        num_passages = st.slider("Görüntülenecek Pasaj Sayısı", min_value=1, max_value=len(passages), value=7, step=1)

        # Generate questions from the selected number of passages
        for idx, passage in enumerate(passages[:num_passages]):  # Limit to the number of selected passages
            questions = generate_questions_pipeline(passage)
            st.subheader(f"Passage {idx + 1}")
            st.write(passage)
            st.write("Generated Questions:")
            for q in questions:
                st.write(f"- {q}")
            st.write(f"\n{'-' * 50}\n")

# PDF Belge Soru Cevaplama Sayfası
elif page == "PDF Belge Soru Cevaplama":
    st.title("PDF Belge Soru Cevaplama")

    uploaded_pdf = st.file_uploader("PDF dosyasını yükleyin", type="pdf")

    if uploaded_pdf is not None:
        # PDF'den metin çıkar
        extracted_text = extract_text_from_pdf(uploaded_pdf)

        # Split the text into sentences
        sentences = sent_tokenize(extracted_text)

        # Combine sentences into passages
        passages = []
        current_passage = ""
        for sentence in sentences:
            if len(current_passage.split()) + len(sentence.split()) < 200:  # adjust the word limit as needed
                current_passage += " " + sentence
            else:
                passages.append(current_passage.strip())
                current_passage = sentence
        if current_passage:
            passages.append(current_passage.strip())

        # Slider to allow the user to select the number of passages to process
        num_passages = st.slider("Görüntülenecek Pasaj Sayısı", min_value=1, max_value=len(passages), value=7, step=1)

        # Load the QA pipeline
        qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

        # Function to track and answer only unique questions
        def answer_unique_questions(passages, qa_pipeline):
            answered_questions = set()  # to store unique questions

            for idx, passage in enumerate(passages[:num_passages]):  # Only process the selected number of passages
                questions = generate_questions_pipeline(passage)

                for question in questions:
                    if question not in answered_questions:  # check if the question has already been answered
                        answer = qa_pipeline({'question': question, 'context': passage})
                        st.write(f"Passage {idx + 1}")
                        st.write(f"Q: {question}")
                        st.write(f"A: {answer['answer']}\n")
                        answered_questions.add(question)  # add the question to the set to avoid repetition
                st.write(f"{'=' * 50}\n")

        # Answer unique questions
        answer_unique_questions(passages, qa_pipeline)

