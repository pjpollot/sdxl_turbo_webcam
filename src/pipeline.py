import torch

from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler
from accelerate import Accelerator
from PIL import Image

class TurboPipeline:
    def __init__(self, model_path_or_url: str, controlnet_path_or_url: str, variant: str | None = None) -> None:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path_or_url,
            torch_dtype=torch.float16,
            variant=variant,
        )
        self._pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_path_or_url,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant=variant,
        )
        # add to fastest device
        device = Accelerator().device
        print(f"Device {device} detected.")
        self._pipeline.to(device)
        # use EulerAncestralDiscreteScheduler
        self._pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(self._pipeline.config, timestep_spacing="trailing")
        # compile the unet for fast inference
        if device == "cuda":
            print("Compile unet.")
            self._pipeline.unet = torch.compile(self._pipeline.unet, mode="reduce-overhead", fullgraph=True)


    def __call__(self, prompt: str, image: Image, overall_strength: float = 7.0, num_inference_steps: int = 1, controlnet_conditioning_scale: float = 0.8) -> Image:
        strength = overall_strength / num_inference_steps
        return self._pipeline(
            prompt=prompt,
            image=image,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            strength=strength,
            guidance_scale=0.0,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images[0]