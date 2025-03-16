import streamlit as st
import whisper
from googletrans import Translator
import googletrans
from gtts import gTTS
from io import BytesIO
import os
import tempfile
from PIL import Image
import easyocr
import cv2
import numpy as np
import asyncio
import time

# Page configuration
st.set_page_config(
    page_title="Polyglot - Multilingual Communication Suite",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #424242;
        margin-bottom: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.7rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        margin-bottom: 1.5rem;
    }
    .action-button {
        background-color: #1E88E5;
        color: white;
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
    }
    .stProgress .st-eb {
        background-color: #1E88E5;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F0F2F6;
        border-radius: 4px 4px 0 0;
        border: none;
        padding: 10px 16px;
        color: #424242;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    .stMarkdown a {
        color: #1E88E5;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Initialize app states
if "transcription" not in st.session_state:
    st.session_state.transcription = ""
if "translated_text" not in st.session_state:
    st.session_state.translated_text = ""
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "translated_ocr_text" not in st.session_state:
    st.session_state.translated_ocr_text = ""
if "progress" not in st.session_state:
    st.session_state.progress = 0
if "processing" not in st.session_state:
    st.session_state.processing = False

# App Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="main-header">🌍 Polyglot</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        Your all-in-one solution for transcription, translation, and text extraction
    </div>
    """, unsafe_allow_html=True)

# Sidebar with improved UI
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=Polyglot", width=150)
    st.markdown("### Settings")
    
    with st.expander("Translation Settings", expanded=True):
        language = st.selectbox(
            "Target Language", 
            options=list(googletrans.LANGUAGES.keys()),
            format_func=lambda x: f"{googletrans.LANGUAGES[x].capitalize()} ({x})",
            index=list(googletrans.LANGUAGES.keys()).index('en')
        )
    
    with st.expander("OCR Settings", expanded=True):
        available_languages = ['en', 'es', 'fr', 'de', 'zh', 'ja', 'ko', 'ar', 'hi']
        language_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 
            'de': 'German', 'zh': 'Chinese', 'ja': 'Japanese',
            'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi'
        }
        
        ocr_languages = st.multiselect(
            "OCR Languages", 
            options=available_languages,
            format_func=lambda x: language_names.get(x, x.capitalize()),
            default=['en']
        )
    
    st.markdown("### About")
    st.markdown("""
    Polyglot helps break language barriers with AI-powered 
    translation and transcription tools.
    
    **Features:**
    - Audio transcription
    - Text translation
    - OCR text extraction
    - Text-to-speech conversion
    """)

# Initialize models (with loading spinners)
@st.cache_resource(show_spinner=False)
def load_whisper_model():
    with st.spinner("Loading speech recognition model..."):
        return whisper.load_model("base")

@st.cache_resource(show_spinner=False)
def load_ocr_reader(languages):
    with st.spinner("Loading OCR model..."):
        return easyocr.Reader(languages if languages else ['en'])

whisper_model = load_whisper_model()
translator = Translator()

# Navigation
icons = {"Audio Transcription": "🎤", "Image OCR": "📄", "Help": "❓"}
selected_tab = st.radio(
    "Choose functionality:",
    ["Audio Transcription", "Image OCR", "Help"],
    format_func=lambda x: f"{icons[x]} {x}",
    horizontal=True
)

# Helper functions
def transcribe_audio(audio_file):
    """Transcribe audio using Whisper."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        result = whisper_model.transcribe(temp_audio.name)
        os.remove(temp_audio.name)
    return result['text']

def text_to_speech(text, lang='en'):
    """Convert text to speech using gTTS."""
    tts = gTTS(text=text, lang=lang)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    return audio_bytes.getvalue()

def draw_boxes(image, results):
    """Draw bounding boxes around detected text."""
    image_np = np.array(image)
    for (bbox, text, prob) in results:
        # Unpack the bounding box
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        
        # Draw the bounding box
        cv2.rectangle(image_np, top_left, bottom_right, (0, 0, 255), 2)
        
        # Add text label with confidence score
        label = f"{text} ({prob:.2f})"
        cv2.putText(
            image_np, label, (top_left[0], top_left[1] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2
        )
    return image_np

def simulate_progress():
    """Simulate progress for better user experience."""
    if st.session_state.processing:
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        st.session_state.processing = False
        return progress_bar
    return None

# Tab 1: Audio Transcription
if selected_tab == "Audio Transcription":
    st.markdown('<div class="sub-header">🎤 Audio Transcription and Translation</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Upload Audio")
        st.write("Supported formats: WAV, MP3, M4A")
        
        uploaded_audio = st.file_uploader(
            "Drag and drop your audio file here",
            type=["wav", "mp3", "m4a"],
            key="audio_uploader"
        )
        
        if uploaded_audio:
            st.audio(uploaded_audio, format=f"audio/{uploaded_audio.name.split('.')[-1]}")
            
            if st.button("🔍 Transcribe Audio", key="transcribe_btn"):
                st.session_state.processing = True
                progress_bar = simulate_progress()
                
                try:
                    st.session_state.transcription = transcribe_audio(uploaded_audio)
                    st.success("✅ Transcription complete!")
                except Exception as e:
                    st.error(f"Error during transcription: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Results")
        
        tabs = st.tabs(["Transcription", "Translation", "Audio Output"])
        
        with tabs[0]:
            if st.session_state.transcription:
                st.text_area(
                    "Original Text:",
                    st.session_state.transcription,
                    height=200
                )
            else:
                st.info("Transcribed text will appear here after processing.")
        
        with tabs[1]:
            if st.session_state.transcription:
                if st.button("🌍 Translate Text", key="translate_btn"):
                    st.session_state.processing = True
                    progress_bar = simulate_progress()
                    
                    try:
                        translation = asyncio.run(translator.translate(
                            st.session_state.transcription, dest=language
                        ))
                        st.session_state.translated_text = translation.text
                        st.success("✅ Translation complete!")
                    except Exception as e:
                        st.error(f"Error during translation: {str(e)}")
            
            if st.session_state.translated_text:
                st.text_area(
                    f"Translated to {googletrans.LANGUAGES[language].capitalize()}:",
                    st.session_state.translated_text,
                    height=200
                )
            else:
                st.info("Translated text will appear here after processing.")
        
        with tabs[2]:
            if st.session_state.translated_text:
                try:
                    audio_output = text_to_speech(st.session_state.translated_text, language)
                    st.audio(audio_output, format="audio/mp3")
                    st.download_button(
                        label="Download Audio",
                        data=audio_output,
                        file_name=f"translated_audio_{language}.mp3",
                        mime="audio/mp3"
                    )
                except Exception as e:
                    st.error(f"Error generating speech: {str(e)}")
            else:
                st.info("Audio output will be available after translation.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Image OCR
elif selected_tab == "Image OCR":
    st.markdown('<div class="sub-header">📄 Image OCR and Translation</div>', unsafe_allow_html=True)
    
    reader = load_ocr_reader(ocr_languages)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Upload Image")
        st.write("Supported formats: JPG, PNG, JPEG")
        
        uploaded_image = st.file_uploader(
            "Drag and drop your image file here",
            type=["jpg", "jpeg", "png"],
            key="image_uploader"
        )
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Extract Text", key="extract_btn"):
                st.session_state.processing = True
                progress_bar = simulate_progress()
                
                try:
                    image_np = np.array(image)
                    results = reader.readtext(image_np)
                    st.session_state.extracted_text = " ".join([result[1] for result in results])
                    
                    # Store the image with boxes for display later
                    if results:
                        st.session_state.image_with_boxes = draw_boxes(image, results)
                    
                    st.success("✅ Text extraction complete!")
                except Exception as e:
                    st.error(f"Error during text extraction: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Results")
        
        tabs = st.tabs(["Extracted Text", "Text Detection", "Translation", "Audio Output"])
        
        with tabs[0]:
            if st.session_state.extracted_text:
                st.text_area(
                    "Extracted Text:",
                    st.session_state.extracted_text,
                    height=150
                )
            else:
                st.info("Extracted text will appear here after processing.")
        
        with tabs[1]:
            if hasattr(st.session_state, 'image_with_boxes'):
                st.image(
                    st.session_state.image_with_boxes,
                    caption="Text Detection Visualization",
                    use_container_width=True
                )
            else:
                st.info("Text detection visualization will appear here after processing.")
        
        with tabs[2]:
            if st.session_state.extracted_text:
                if st.button("🌍 Translate Extracted Text", key="translate_ocr_btn"):
                    st.session_state.processing = True
                    progress_bar = simulate_progress()
                    
                    try:
                        translation = asyncio.run(translator.translate(
                            st.session_state.extracted_text, dest=language
                        ))
                        st.session_state.translated_ocr_text = translation.text
                        st.success("✅ Translation complete!")
                    except Exception as e:
                        st.error(f"Error during translation: {str(e)}")
            
            if st.session_state.translated_ocr_text:
                st.text_area(
                    f"Translated to {googletrans.LANGUAGES[language].capitalize()}:",
                    st.session_state.translated_ocr_text,
                    height=150
                )
            else:
                st.info("Translated text will appear here after processing.")
        
        with tabs[3]:
            if st.session_state.translated_ocr_text:
                try:
                    audio_output = text_to_speech(st.session_state.translated_ocr_text, language)
                    st.audio(audio_output, format="audio/mp3")
                    st.download_button(
                        label="Download Audio",
                        data=audio_output,
                        file_name=f"ocr_translated_audio_{language}.mp3",
                        mime="audio/mp3"
                    )
                except Exception as e:
                    st.error(f"Error generating speech: {str(e)}")
            else:
                st.info("Audio output will be available after translation.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Help
elif selected_tab == "Help":
    st.markdown('<div class="sub-header">❓ Help & Documentation</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Getting Started")
    st.markdown("""
    Welcome to Polyglot! This app helps you break language barriers with AI-powered transcription, translation, and OCR capabilities.
    
    **Quick Start Guide:**
    1. Choose a feature from the navigation bar (Audio Transcription or Image OCR)
    2. Upload your file (audio or image)
    3. Process the file (transcribe or extract text)
    4. Translate the extracted text to your target language
    5. Generate and download speech from the translated text
    
    For best results, use clear audio recordings and high-quality images with legible text.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Audio Transcription Tips")
        st.markdown("""
        - **Supported formats:** WAV, MP3, M4A
        - **Best audio quality:** Clear speech, minimal background noise
        - **Recommended duration:** 5 seconds to 10 minutes
        - **Languages:** Multiple languages supported via Whisper model
        
        **Troubleshooting:**
        - If transcription is inaccurate, try reducing background noise
        - For long files, allow extra processing time
        - If you encounter errors, try a different audio format
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Image OCR Tips")
        st.markdown("""
        - **Supported formats:** JPG, PNG, JPEG
        - **Best image quality:** High resolution, good lighting, clear contrast
        - **OCR languages:** Select appropriate language(s) for your text
        - **Text styles:** Works with printed text and some handwriting
        
        **Troubleshooting:**
        - If text detection fails, try improving image contrast
        - For complex layouts, results may vary
        - Multi-language documents may require multiple language selections
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Frequently Asked Questions")
    
    with st.expander("What languages are supported for translation?"):
        st.write("Polyglot supports translation to and from over 100 languages using Google Translate's API.")
    
    with st.expander("How accurate is the audio transcription?"):
        st.write("The app uses OpenAI's Whisper model which provides good accuracy for clear speech. Performance may vary with accents, background noise, and audio quality.")
    
    with st.expander("Can I process handwritten text?"):
        st.write("Yes, EasyOCR can detect some handwritten text, but performance is best with printed text. Results depend on handwriting clarity and image quality.")
    
    with st.expander("Is there a file size limit?"):
        st.write("Streamlit has a default file size limit of 200MB, but we recommend keeping audio files under 10MB and images under 5MB for optimal performance.")
    
    with st.expander("How can I improve the translation quality?"):
        st.write("Ensure the transcription or text extraction is accurate first. Clear, grammatically correct source text leads to better translations.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f0f2f6; border-radius: 0.7rem;">
    <p>Created with ❤️ using Streamlit, Whisper, EasyOCR, and Google Translate</p>
    <p>© 2025 Polyglot - Breaking language barriers with AI</p>
</div>
""", unsafe_allow_html=True)
