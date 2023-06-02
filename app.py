import gradio as gr
from huggingface_hub import PyTorchModelHubMixin
import torch
import matplotlib.pyplot as plt
import torchvision
from networks_fastgan import MyGenerator
import click
import PIL
from image_generator import generate_images
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
import cv2
import sys
import numpy as np
sys.path.append('Real-ESRGAN')
from realesrgan import RealESRGANer

import os


def image_generation(model, number_of_images=1):
    img = generate_images(model)
    #TODO: run this image through the ESRGAN upscaler and return it, simple enough ?
    upscaled_img = torchvision.transforms.functional.resize(img, (1024, 1024), interpolation=2)
    upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    #model_path = load_file_from_url(url=file_url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
    model_path = os.path.join('weights', 'RealESRGAN_x4plus.pth')
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        dni_weight=None,
        model=upscale_model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
  )
    #TODO: img has to be same as opencv imread format
    open_cv_image = np.array(img) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    #print(type(open_cv_image)) 
    #print(type(img))
    #print(type(upscaled_img))
    output, _ = upsampler.enhance(open_cv_image, outscale=4)
    output2, _ = upsampler.enhance(output   , outscale=4)
    #return f"generating {number_of_images} images from {model}"
    cv2.imwrite('out/output_upscaled.png', output)
    cv2.imwrite('out/output_upscaled_dupli.png', output2)
    cv2.imwrite('out/output.png', np.array(img)[:, :, ::-1])
    output2 = cv2.cvtColor(output2, cv2.COLOR_BGR2RGB)
    return PIL.Image.fromarray(output2)
if __name__ == "__main__":
    description = "This is a web demo of a projected GAN trained on photos of thirty paintings from the series of paintings Welcome home.                                                                                         The abstract expressionism and color field models were initially trained on images from their perspective art directions and then transfer learned to Hana's houses."
    inputs = gr.inputs.Radio(["Hana Hanak houses", "Hana Hanak houses - abstract expressionism", "Hana Hanak houses - color field"])
    outputs = gr.outputs.Image(label="Generated Image", type="pil")
    #outputs = "text"
    title = "Anti house generator"
    article = "<p style='text-align: center'><a href='https://github.com/autonomousvision/projected_gan'>Official projected GAN github repo + paper</a></p>"



    demo = gr.Interface(image_generation, inputs, outputs, title=title, article = article, description = description, 
    analytics_enabled=False)
    demo.launch(share=True)

    #app, local_url, share_url = iface.launch(share=True)
