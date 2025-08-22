import streamlit as st
import subprocess
import os
from transformers import pipeline

# Load Whisper model (multilingual)
@st.cache_resource
def load_asr_model():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        stride_length_s=5,
    )

# Function to extract audio from video
def extract_audio(video_path, audio_path="temp_audio.wav"):
    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",  # remove video
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

# Streamlit app
def main():
    st.title("🎥 Video to Text (Choose English or French)")

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

            st.success("✅ Completed!")

            st.subheader(f"{language_choice} Transcription")
            st.text_area(f"{language_choice} Text", transcript_text, height=300)

            # Download button
            st.download_button(
                f"Download {language_choice} Transcription",
                transcript_text,
                file_name=f"transcription_{language_choice.lower()}.txt"
            )

            # Cleanup
            os.remove(audio_path)
            os.remove(temp_video_path)

if __name__ == "__main__":
    main()
