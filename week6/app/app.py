import io

import torch
import torch.backends.mps as mps
import uvicorn
from diffusers import StableDiffusionPipeline
from fastapi import FastAPI
from starlette.responses import StreamingResponse

app = FastAPI()


def get_device():
    if mps.is_available():
        print("Mps is available!")
        return torch.device("mps")
    else:
        print("Mps is not available!")
        return torch.device("cpu")


@app.post("/imaggen")
def imggen(prompt: str):
    device = get_device()
    pipe = StableDiffusionPipeline.from_pretrained("../model")
    pipe = pipe.to(device)

    img = pipe(prompt).images[0]

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    del pipe

    return StreamingResponse(img_byte_arr, media_type="image/png")


# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
