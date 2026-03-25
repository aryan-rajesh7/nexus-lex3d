import torch
from diffusers import StableDiffusionPipeline, ShapEPipeline, AudioLDMPipeline, StableVideoDiffusionPipeline

def download_models():
    try:
        StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    except Exception as e:
        print(f"Warning on SD: {e}")

    try:
        ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16)
    except Exception as e:
        print(f"Warning on Shap-E: {e}")

    try:
        AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2", torch_dtype=torch.float16)
    except Exception as e:
        print(f"Warning on Audio: {e}")
    
    try:
        StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
    except Exception as e:
        print(f"Warning on SVD: {e}")

if __name__ == "__main__":
    download_models()
