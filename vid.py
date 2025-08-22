import streamlit as st
import subprocess
import os
from transformers import pipeline
import fitz  # PyMuPDF for PDF text extraction

# Load Whisper model (for transcription)
@st.cache_resource
def load_asr_model():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        stride_length_s=5,
    )

# Load summarization model
@st.cache_resource
def load_summary_model():
    return pipeline("summarization", model="t5-small")

# Extract audio from video using ffmpeg
def extract_audio(video_path, audio_path="temp_audio.wav"):
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
        "-y",
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return audio_path

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(pdf_file)
    for page in doc:
        text += page.get_text()
    return text

# Streamlit App
def main():
    st.title("📄 PDF Summarizer + 🎥 Video Transcriber")

    mode = st.radio("Choose Mode:", ["Video Transcription", "PDF Summarization"])

    if mode == "Video Transcription":
        uploaded_video = st.file_uploader(
            "Upload a video", type=["mp4", "mkv", "mov", "avi"]
        )

        # Language selection
        language_choice = st.radio(
            "Select Transcription Language:",
            ("English", "French")
        )

        if uploaded_video:
            temp_video_path = f"temp_{uploaded_video.name}"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            st.video(temp_video_path)

            if st.button("Transcribe"):
                with st.spinner("Extracting audio..."):
                    audio_path = extract_audio(temp_video_path)

                with st.spinner(f"Transcribing in {language_choice}..."):
                    asr_pipeline = load_asr_model()
                    transcription = asr_pipeline(
                        audio_path,
                        generate_kwargs={
                            "language": "english" if language_choice == "English" else "french"
                        }
                    )
                    transcript_text = transcription["text"]

                st.success("✅ Transcription Completed!")

                st.subheader(f"{language_choice} Transcription")
                st.text_area(f"{language_choice} Text", transcript_text, height=300)

                st.download_button(
                    f"Download {language_choice} Transcription",
                    transcript_text,
                    file_name=f"transcription_{language_choice.lower()}.txt"
                )

                os.remove(audio_path)
                os.remove(temp_video_path)

    elif mode == "PDF Summarization":
        uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

        if uploaded_pdf:
            temp_pdf_path = f"temp_{uploaded_pdf.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())

            if st.button("Summarize PDF"):
                with st.spinner("Extracting text from PDF..."):
                    pdf_text = extract_text_from_pdf(temp_pdf_path)

                with st.spinner("Summarizing text..."):
                    summarizer = load_summary_model()
                    summary = summarizer(
                        pdf_text,
                        max_length=150,
                        min_length=50,
                        do_sample=False
                    )[0]["summary_text"]

                st.success("✅ PDF Summarization Completed!")

                st.subheader("PDF Summary")
                st.write(summary)

                st.download_button(
                    "Download PDF Summary",
                    summary,
                    file_name="pdf_summary.txt"
                )

                os.remove(temp_pdf_path)


if __name__ == "__main__":
    main()
