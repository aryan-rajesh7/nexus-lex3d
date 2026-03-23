import warnings
# Silence "CUDA" warnings on Mac
warnings.filterwarnings("ignore", category=UserWarning, message="User provided device_type of 'cuda'")

import torch
from diffusers import (
    StableDiffusionPipeline, 
    ShapEPipeline, 
    AudioLDMPipeline, 
    StableVideoDiffusionPipeline,
    DPMSolverMultistepScheduler
)
# We use export_to_video for MP4s now
from diffusers.utils import export_to_gif, export_to_video
import gc
import os

class SynaesthesiaCore:
    def __init__(self):
        # Detect Mac Apple Silicon
        if torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.pipe = None
        self.current_mode = None

    def clear_memory(self):
        """Aggressively clear RAM between modality switches"""
        if self.pipe:
            del self.pipe
        self.pipe = None
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()

    # --- Modality 1: Text to Image ---
    def generate_image(self, prompt):
        if self.current_mode != "t2i":
            self.clear_memory()
            
            # Use float32 to prevent black images/crashes
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                torch_dtype=torch.float32, 
                use_safetensors=True,
                variant=None 
            )
            
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.safety_checker = None 
            pipe.enable_attention_slicing()
            
            self.pipe = pipe.to(self.device)
            self.current_mode = "t2i"
        
        result = self.pipe(prompt, num_inference_steps=20)
        image = result.images[0]
        return image

    # --- Modality 2: Image to 3D (Balanced Mode) ---
    def generate_3d(self, prompt):
        if self.current_mode != "3d":
            self.clear_memory()
            
            pipe = ShapEPipeline.from_pretrained(
                "openai/shap-e", 
                torch_dtype=torch.float32, 
                use_safetensors=False, 
                variant=None 
            )
            
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing()
            
            self.pipe = pipe
            self.current_mode = "3d"
        
        # 96px is sharper than 64px, but faster than 128px.
        images = self.pipe(
            prompt, 
            guidance_scale=15.0, 
            num_inference_steps=32, 
            frame_size=96 
        ).images
        
        gif_path = "output_3d.gif"
        export_to_gif(images[0], gif_path)
        return gif_path

    # --- Modality 3: Text to Music ---
    def generate_music(self, prompt, duration=5):
        if self.current_mode != "audio":
            self.clear_memory()
            
            pipe = AudioLDMPipeline.from_pretrained(
                "cvssp/audioldm-s-full-v2", 
                torch_dtype=torch.float32,
                variant=None
            )
            self.pipe = pipe.to(self.device)
            self.current_mode = "audio"

        audio = self.pipe(prompt, num_inference_steps=10, audio_length_in_s=duration).audios[0]
        return audio, 16000

   # --- Modality 4: Image to Video ---
    def generate_video(self, image):
        if self.current_mode != "video":
            self.clear_memory()
            
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt", 
                torch_dtype=torch.float32, 
                variant=None 
            )
            
            # This ensures the model weights move to the GPU exactly when needed
            pipe.enable_sequential_cpu_offload()
            
            self.pipe = pipe
            self.current_mode = "video"
        
        # Resize to standard SVD resolution
        image = image.resize((576, 320))
        
        frames = self.pipe(
            image, 
            decode_chunk_size=2, 
            num_frames=7, 
            motion_bucket_id=127, 
            generator=torch.manual_seed(42),
            height=320,
            width=576
        ).frames[0]
        
        path = "output_video.mp4"
        export_to_video(frames, path, fps=7)
        return path