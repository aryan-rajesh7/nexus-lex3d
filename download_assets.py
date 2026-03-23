import torch
from diffusers import StableDiffusionPipeline, ShapEPipeline, AudioLDMPipeline, StableVideoDiffusionPipeline

def download_models():
    print("⬇️ Downloading SynaesthAI Assets... This may take a while.")
    
    # 1. Text -> Image
    print("1/4 Downloading Stable Diffusion...")
    try:
        StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    except Exception as e:
        print(f"Warning on SD: {e}")

    # 2. Image -> 3D
    print("2/4 Downloading Shap-E...")
    try:
        ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16)
    except Exception as e:
        print(f"Warning on Shap-E: {e}")

    # 3. Text -> Music
    print("3/4 Downloading AudioLDM...")
    try:
        AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2", torch_dtype=torch.float16)
    except Exception as e:
        print(f"Warning on Audio: {e}")
    
    # 4. Image -> Video
    print("4/4 Downloading Stable Video Diffusion...")
    try:
        StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
    except Exception as e:
        print(f"Warning on SVD: {e}")

    print("✅ All assets downloaded.")

if __name__ == "__main__":
    download_models()
