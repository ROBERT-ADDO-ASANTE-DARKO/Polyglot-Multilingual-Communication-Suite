# 🌍 Polyglot: Multilingual Communication Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

Polyglot is an end-to-end solution for breaking language barriers using AI-powered transcription, translation, and text extraction capabilities. Built with Streamlit, this application provides an intuitive interface for multilingual communication.

## 🚀 Features

- **Audio Transcription**: Convert spoken language to text using OpenAI's Whisper model
- **Text Translation**: Translate between 100+ languages via Google Translate API
- **Image OCR**: Extract text from images and documents using EasyOCR
- **Text-to-Speech**: Convert translated text to speech with gTTS
- **Modern UI**: Intuitive, card-based interface with proper navigation and progress indicators

## 📋 Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## 🔧 Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/polyglot.git
cd polyglot
```

Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Audio Transcription
1. Navigate to the "Audio Transcription" tab
2. Upload an audio file (WAV, MP3, M4A)
3. Click "Transcribe Audio"
4. View transcription and click "Translate Text" if needed
5. Listen to or download the translated speech

### Image OCR
1. Navigate to the "Image OCR" tab
2. Upload an image file (JPG, PNG, JPEG)
3. Click "Extract Text"
4. View extracted text and click "Translate Text" if needed
5. Listen to or download the translated speech

## 🛠️ Technologies Used

- [Streamlit](https://streamlit.io/) - Frontend framework
- [Whisper](https://github.com/openai/whisper) - Speech recognition model
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Optical Character Recognition
- [Google Translate](https://pypi.org/project/googletrans/) - Translation API
- [gTTS](https://pypi.org/project/gTTS/) - Text-to-Speech conversion

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📬 Contact

Your Name - [@yourtwitterhandle](https://twitter.com/yourtwitterhandle) - raddo2961@gmail.com

Project Link: [https://github.com/yourusername/polyglot]([https://github.com/yourusername/polyglot](https://github.com/ROBERT-ADDO-ASANTE-DARKO/Polyglot-Multilingual-Communication-Suite))

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) for the Whisper model
- [JaidedAI](https://github.com/JaidedAI) for EasyOCR
- All contributors and open-source libraries that made this project possible
