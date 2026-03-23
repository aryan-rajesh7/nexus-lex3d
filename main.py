import torch
from diffusers import StableDiffusionPipeline, ShapEPipeline, AudioLDMPipeline, StableVideoDiffusionPipeline

def download_models():

    StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

    ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16)

    AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2", torch_dtype=torch.float16)
    
    StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")

if __name__ == "__main__":
    download_models()
