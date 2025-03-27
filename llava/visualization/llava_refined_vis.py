"""
LLaVA Refined Visualizer

Author: Nicholas Mesa-Cucalon

Description: This script will use ideas from 
https://github.com/zjysteven/VLM-Visualizer
to try and visualize each attention head of LLaVA.
"""

# Imports
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch
import torch.nn.functional as F

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

"""
Step 1: Model Setup
"""
model_path = "liuhaotian/llava-v1.5-7b"
load_8bit = False
load_4bit = True
device = "cuda" if torch.cuda.is_available() else "cpu"
# Disables redundant Torch initializations
disable_torch_init() 
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, 
    None, # model_base
    model_name, 
    load_8bit, 
    load_4bit, 
    device=device
)
# Setup image and prompt
url = "https://images.contentstack.io/v3/assets/blt6f84e20c72a89efa/blt0931485f70d1f7d6/64023bbde70dd635488d05cb/article-cats-fighting-or-playing-header@1.5x.
prompt_text = "What is happening in this image?"
# Conversation Type Setup
if "llama-2" in model_name.lower():
    conv_mode = "llava_llama_2"
elif "mistral" in model_name.lower():
    conv_mode = "mistral_instruct"
elif "v1.6-34b" in model_name.lower():
    conv_mode = "chatml_direct"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
elif "mpt" in model_name.lower():
    conv_mode = "mpt"
else:
    conv_mode = "llava_v0"
conv = conv_templates[conv_mode].copy()
if "mpt" in model_name.lower():
    roles = ('user', 'assistant')
else:
    roles = conv.roles
# Image Setup
image = load_image(image_path_or_url)
image_tensor, images = process_images([image], image_processor, model.config)
image = images[0]
image_size = image.size
if type(image_tensor) is list:
    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
else:
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text
conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# NOTE: We manually change the prompt here or 
# the system prompt will overpower the attn mechanism
