from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import urllib.parse as parse
import os
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = f"./vit-base-catjig/checkpoint-1000"

model = ViTForImageClassification.from_pretrained(model_name).to(device)

image_processor = ViTImageProcessor.from_pretrained(model_name)


def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False


def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)


def get_prediction(model, url_or_path):
    # load the image
    img = load_image(url_or_path)
    # preprocessing the image
    pixel_values = image_processor(img, return_tensors="pt")["pixel_values"].to(device)
    # perform inference
    output = model(pixel_values)
    # get the label id and return the class name
    return model.config.id2label[int(output.logits.softmax(dim=1).argmax())]


print(get_prediction(model, "./puzzles/test/3-6-7-11-1-4-9-5-8-0-10-14-12-2-13-15/cat-00000237-022.jpg"))
