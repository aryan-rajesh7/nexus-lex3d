Nexus and Lex3D

Nexus and Lex3D is a cross-modal AI generation suite that allows you to:

- Generate images from text prompts (Stable Diffusion)
- Generate 3D assets from text prompts (OpenAI Shap-E)
- Generate videos from images (Stable Video Diffusion)
- Generate music from text prompts (AudioLDM)

---

Features

1. Text to Image: Generate high-quality images from descriptive prompts
2. Image to Video: Animate generated images into short video clips
3. Text to Music: Compose audio tracks based on textual prompts
4. Text to 3D: Generate 3D assets (rotating GIFs) from textual descriptions

---

Installation

git clone <your-repo-url>
cd nexus-lex3d
pip install -r requirements.txt

---

Usage

streamlit run main.py

- Use the sidebar to select Nexus Multimedia AI Generation System or Lex3D Text-to-3D Asset Generator
- Enter your prompts and generate content
- All models are downloaded at runtime; no model weights are included in this repository

---

Models are downloaded at runtime from their official sources. Users must comply with the respective model licenses.

---
