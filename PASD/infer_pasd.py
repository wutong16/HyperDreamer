import os
import sys
import cv2
import glob
import argparse
import numpy as np
from PIL import Image
import safetensors.torch

import torch
from torchvision import transforms
import torch.utils.checkpoint

from diffusers import AutoencoderKL, PNDMScheduler, LCMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler#, StableDiffusionControlNetPipeline
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from os import path
sys.path.append(path.dirname(path.abspath(__file__)))

from pipelines.pipeline_pasd import StableDiffusionControlNetPipeline
from myutils.misc import load_dreambooth_lora
from myutils.wavelet_color_fix import wavelet_color_fix
#from annotator.retinaface import RetinaFaceDetection


class PASD_args:
    def __init__(self):
        self.pretrained_model_path = 'PASD/checkpoints/stable-diffusion-v1-5'
        self.lcm_lora_path ='PASD/checkpoints/lcm-lora-sdv1-5'
        self.pasd_model_path = 'PASD/runs/pasd/checkpoint-100000'
        self.personalized_model_path = None
        self.control_type = 'realisr'
        self.high_level_info = 'caption' # TODO, & prompt
        self.prompt = ''
        self.added_prompt = 'clean, high-resolution, 8k'
        self.negative_prompt = 'blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed'
        self.input_image = None # PIL Image
        self.mixed_precision = 'fp16'
        self.guidance_scale = 9.0
        self.conditioning_scale = 1.0
        self.blending_alpha = 1.0
        self.multiplier = 0.6
        self.num_inference_steps = 20
        self.process_size = 768
        self.decoder_tiled_size = 224
        self.encoder_tiled_size = 1024
        self.latent_tiled_size = 320
        self.latent_tiled_overlap = 8
        self.upscale = 1
        self.use_personalized_model = False
        self.use_pasd_light = False
        self.use_lcm_lora = False
        self.use_blip = False
        self.init_latent_with_noise = False
        self.added_noise_level = 400
        self.offset_noise_scale = 0.0


def load_pasd_pipeline(args, device, enable_xformers_memory_efficient_attention):
    if args.use_pasd_light:
        from models.pasd_light.unet_2d_condition import UNet2DConditionModel
        from models.pasd_light.controlnet import ControlNetModel
    else:
        from models.pasd.unet_2d_condition import UNet2DConditionModel
        from models.pasd.controlnet import ControlNetModel
    # Load scheduler, tokenizer and models.
    scheduler = UniPCMultistepScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(args.pasd_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.pasd_model_path, subfolder="controlnet")

    personalized_model_root = "checkpoints/personalized_models"
    if args.use_personalized_model and args.personalized_model_path is not None:
        if os.path.isfile(f"{personalized_model_root}/{args.personalized_model_path}"):
            unet, vae, text_encoder = load_dreambooth_lora(unet, vae, text_encoder, f"{personalized_model_root}/{args.personalized_model_path}", 
                                                           blending_alpha=args.blending_alpha, multiplier=args.multiplier)
        else:
            unet = UNet2DConditionModel.from_pretrained_orig(personalized_model_root, subfolder=f"{args.personalized_model_path}") # unet_disney

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    # weight_dtype = torch.float16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    controlnet.to(device, dtype=weight_dtype)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
    #validation_pipeline.enable_vae_tiling()
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.encoder_tiled_size, decoder_tile_size=args.decoder_tiled_size)

    if args.use_lcm_lora:
        # load and fuse lcm lora
        validation_pipeline.load_lora_weights(args.lcm_lora_path)
        validation_pipeline.fuse_lora()
        validation_pipeline.scheduler = LCMScheduler.from_config(validation_pipeline.scheduler.config)

    return validation_pipeline

def load_high_level_net(args, device='cuda'):
    if args.use_blip: # False
        from lavis.models import load_model_and_preprocess
        model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
        return model, vis_processors, None
    else:
        model, _, transform = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14",
            pretrained="mscoco_finetuned_laion2B-s13B-b90k"
            )
        return model, transform, None
    
def get_validation_prompt(args, image, model, preprocess, category, device='cuda'):
    validation_prompt = ""

    if args.use_blip: # False
        image = preprocess["eval"](image).unsqueeze(0).to(device)
        caption = model.generate({"image": image}, num_captions=1)[0]
        caption = caption.replace("blurry", "clear").replace("noisy", "clean") #
        validation_prompt = caption if args.prompt=="" else f"{caption}, {args.prompt}"
    else:
        image = preprocess(image).unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            generated = model.generate(image)
        caption = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
        caption = caption.replace("blurry", "clear").replace("noisy", "clean") #
        validation_prompt = caption if args.prompt=="" else f"{caption} {args.prompt}"

    return validation_prompt

def pasd_main(args, device, enable_xformers_memory_efficient_attention=True,):

    generator = torch.Generator(device=device)

    pipeline = load_pasd_pipeline(args, device, enable_xformers_memory_efficient_attention)
    model, preprocess, category = load_high_level_net(args, device)

    resize_preproc = transforms.Compose([
        transforms.Resize(args.process_size, interpolation=transforms.InterpolationMode.BILINEAR),
    ])

    validation_image = args.input_image.convert("RGB")
    validation_prompt = get_validation_prompt(args, validation_image, model, preprocess, category)
    validation_prompt = 'a strawberry in white background' # XXX
    print(validation_prompt)
    validation_prompt += args.added_prompt # clean, extremely detailed, best quality, sharp, clean
    negative_prompt = args.negative_prompt #dirty, messy, low quality, frames, deformed,
    
    # print(validation_prompt)

    ori_width, ori_height = validation_image.size
    resize_flag = False
    rscale = args.upscale

    validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale))

    if min(validation_image.size) < args.process_size:
        validation_image = resize_preproc(validation_image)

    validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8))
    #width, height = validation_image.size
    resize_flag = True #

    try:
        image = pipeline(
                args, validation_prompt, validation_image, num_inference_steps=args.num_inference_steps, generator=generator, #height=height, width=width,
                guidance_scale=args.guidance_scale, negative_prompt=negative_prompt, conditioning_scale=args.conditioning_scale,
            ).images[0]
    except Exception as e:
        print(e)
        exit(-1)

    if args.control_type=="realisr": #args.conditioning_scale < 1.0:
        image = wavelet_color_fix(image, validation_image)

    if args.control_type=="realisr" and resize_flag: 
        image = image.resize((ori_width*rscale, ori_height*rscale))

    return image
