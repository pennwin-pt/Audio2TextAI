# Audio2TextAI - AI-Powered Speech-to-Text Service

**Audio2TextAI** is a Python-based tool that converts WAV audio files to text using AI models (e.g., Whisper, Wav2Vec). It provides a RESTful API for seamless integration with other applications.

---

## 🌟 Features
- **High Accuracy**: Leverages AI models for precise transcription.
- **Easy Integration**: RESTful API for sending audio files and receiving text.
- **Extensible**: Supports future audio formats (e.g., MP3, FLAC) and model upgrades.
- **Lightweight**: Built with Flask for quick deployment.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
git clone https://github.com/pennwin-pt/Audio2TextAI.git
cd Audio2TextAI
pip install -r requirements.txt
```

### 2. Configure Model
Edit config.py to set the model path:
```python
MODEL_PATH = "path/to/your/whisper-model"
```

### 3. Run the Service
```bash
python main.py
```