import streamlit as st
from logic import SynaesthesiaCore
import numpy as np
from scipy.io.wavfile import write

st.set_page_config(page_title="Nexus and Lex3D", layout="wide", page_icon="🧠")

if "engine" not in st.session_state:
    st.session_state.engine = SynaesthesiaCore()

st.sidebar.title("Nexus and Lex3D")
st.sidebar.info("Offline Neural Suite running on local hardware.")
mode = st.sidebar.radio("Select Project Module", 
    ["Nexus Multimedia AI Generation System", "Lex3D Text-to-3D Asset Generator"])

if mode == "Nexus Multimedia AI Generation System":
    st.title("Nexus")
    st.markdown("### Cross-Modal Generation Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Text Input")
        prompt = st.text_area("Describe your dream:", "A cyberpunk city with neon lights and flying cars")
        
        if st.button("Generate Image"):
            with st.spinner("Dreaming... (Stable Diffusion)"):
                img = st.session_state.engine.generate_image(prompt)
                st.session_state.last_image = img
                
                if "video_path" in st.session_state:
                    del st.session_state["video_path"]
                
                st.success("Image Generated!")
    
    with col2:
        if "last_image" in st.session_state:
            st.image(st.session_state.last_image, caption="Generated Vision")
            
            st.markdown("---")
            st.subheader("2. Cross-Modality")
            
            
            if "video_path" in st.session_state:
                st.video(st.session_state["video_path"])
                st.success("Video Animation Complete")
                
                
                if st.button("Regenerate Video"):
                    del st.session_state["video_path"]
                    st.rerun()

            else:
                if st.button("Extend to Video (Img2Video)"):
                    with st.spinner("Animating... (Stable Video Diffusion)"):
                        vid_path = st.session_state.engine.generate_video(st.session_state.last_image)
                        
                        st.session_state["video_path"] = vid_path
                        st.rerun()
            
            st.markdown("---")
            
            if st.button("Synthesize Soundtrack (Text2Audio)"):
                with st.spinner("Composing... (AudioLDM)"):
                    audio_data, rate = st.session_state.engine.generate_music(prompt)
                    write("output_music.wav", rate, audio_data.T)
                    st.audio("output_music.wav")

elif mode == "Lex3D Text-to-3D Asset Generator":
    st.title("Lex3D")
    st.markdown("""
    **Technology:** OpenAI Shap-E (Implicit Neural Radiance Fields + Diffusion).
    **Output:** Generates 3D assets from text prompts.
    """)
    
    prompt_3d = st.text_input("Object Description:", "A futuristic robotic helmet, high detail")
    
    if st.button("Forge 3D Model"):
        with st.spinner("Forging NeRF Latents..."):
            gif_path = st.session_state.engine.generate_3d(prompt_3d)
            st.image(gif_path, caption="Rotating 3D Render")
            st.info("Note: To export STL for printing, Python requires 'trimesh' conversion of the latent mesh.")
