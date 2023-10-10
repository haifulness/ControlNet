import boto3
import cv2
import io
import os
import numpy as np
from app.core.config import settings
from app.core.hough2image import process
from annotator.util import resize_image, HWC3
from fastapi import FastAPI
from functools import lru_cache
from loguru import logger
from pathlib import Path
from PIL import Image
from tqdm import tqdm


app = FastAPI(root_path=settings.api_host)
s3 = boto3.resource(
    's3',
    aws_access_key_id=settings.s3_access_key, 
    aws_secret_access_key=settings.s3_secret_key
)
bucket = s3.Bucket(settings.s3_root_dir)


@lru_cache()
def get_settings():
    return settings


@app.get("/healthz")
async def health_check():
    return {"msg": "ok"}


@app.get('/run')
async def run():
    # Download images from S3
    
    # objs = list(bucket.objects.filter(Prefix=settings.s3_input_dir))
    # for obj in tqdm(objs):
    #     filename = obj.key.split('/')[-1]
    #     filepath = 'app/input/' + filename
    #     if len(filename) == 0 or os.path.exists(filepath):
    #         continue
    #     bucket.download_file(obj.key, filepath)

    # Process the images
    
    # np_array = np.zeros((100, 100, 3), np.uint8)
    # im = Image.fromarray(np_array)
    # bits = io.BytesIO()
    # im.save('app/output/_1.png')

    input_image = []
    num_samples = 1
    image_resolution = 256
    strength = 1.0
    guess_mode = False
    detect_resolution = 256
    value_threshold = 0.1
    distance_threshold = 0.1
    ddim_steps = 20
    scale = 9.0
    seed = 42
    eta = 0.0
    prompt = ''
    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    for filepath in Path('app/input').glob('*.*'):
        img = np.array(
            Image.open(filepath.as_posix()).convert('RGB'), 
            dtype=np.uint8,
        )

        output = process(img, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, value_threshold, distance_threshold)
        print(output)


    # Upload output to S3
    
    # for filepath in Path('app/output').glob('*.*'):
    #     filename = filepath.as_posix().split('/')[-1]
    #     bucket.upload_file(
    #         filename, 
    #         settings.s3_output_dir + filename
    #     )

    return {"msg": "done"}


@app.get('/download')
async def download():
    return {}


# uvicorn app.main:app --host 0.0.0.0 --port 80 --reload --reload-dir app
