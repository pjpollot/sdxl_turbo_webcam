import os
import gradio as gr

from argparse import ArgumentParser, Namespace

from src.pipeline import TurboPipeline
from src.preprocessing import square_resize, remove_background, canny_transform

from PIL import Image

error_img_path = os.path.join(
    os.path.dirname(__file__),
    "images/error.png",
)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path_or_url",
        type=str,
        default="stabilityai/sdxl-turbo",
    )
    parser.add_argument(
        "--canny_controlnet_path_or_url",
        type=str,
        default="diffusers/controlnet-canny-sdxl-1.0",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="fp16",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--port",
        type=str,
        required=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipe = TurboPipeline(args.model_path_or_url, args.canny_controlnet_path_or_url, args.variant)

    def image_generation(prompt: str, image: Image, strength: float, num_inference_steps: int, controlnet_strength: float) -> Image:
        if image is not None:
            image = square_resize(image, args.resolution)
            image = remove_background(image)
            image = canny_transform(image)
            return pipe(prompt, image, strength, num_inference_steps, controlnet_strength)
        else:
            print("WARNING: None type image has been given in input.")
            return Image.open(error_img_path)
    
    app = gr.Interface(
        fn=image_generation,
        inputs=[
            gr.Text(),
            gr.Image(type="pil", sources=["webcam"], streaming=True),
            gr.Slider(0, 100, 7.0, step=0.1),
            gr.Slider(1, 20, 1, step=1),
            gr.Slider(0, 1, 0.8, step=0.01),
        ],
        outputs=[
            gr.Image(type="pil"),
        ],
        live=True,
    )
    app.launch(server_port=args.port)
