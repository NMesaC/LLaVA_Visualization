"""
LLaVA + gScoreCam

Author: Nicholas Mesa-Cucalon

Description: This script will use gScoreCam to visualize CLIP's performance when given a text
input. We will also leverage ideas and optimizations from 
https://github.com/zjysteven/VLM-Visualizer/blob/main/llava_example.ipynb
to improve visualization results.
"""

#
"""
Imports
"""

import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

#
"""
Helper Functions
"""
def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


#
"""
Model Setup
"""
disable_torch_init()
image_path        = "https://static.scientificamerican.com/sciam/cache/file/825ED237-152D-4AA0-AB74012B103ECD6C_source.png?w=1350"
prompt_text       = "What is the cat in the image doing?"
model_path        = "liuhaotian/llava-v1.5-7b"
image_file        = "https://llava-vl.github.io/static/images/view.jpg"
load_8bit         = False
load_4bit         = True
model_name        = get_model_name_from_path(model_path)
device            = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, 
    None, # model_base
    model_name, 
    load_8bit, 
    load_4bit, 
    device=device
)
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

#
"""
Input Setup
"""
image = load_image(image_path)
image_tensor = process_images([image], image_processor, model.config)
image_size = image.size
if type(image_tensor) is list:
    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
else:
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

if model.config.mm_use_im_start_end:
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt_text
else:
    inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text
conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
"""
NOTE:
    We remove the system prompt here or else the attention will over focus on the prompt.
    This plus the basic code structure is borrowed from cli.py + the cited github page
"""
prompt = prompt.replace(
    "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. ",
    ""
)
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

#
"""
Setting up gScoreCam for LLaVA CLIP
"""
clip_model = model.get_vision_tower().vision_tower









# #
# """
# Response Generation
# """
# with torch.inference_mode():
#     outputs = model.generate(
#         input_ids,
#         images=image_tensor,
#         image_sizes=[image_size],
#         do_sample=False,
#         max_new_tokens=512,
#         use_cache=True,
#         return_dict_in_generate=True,
#         output_attentions=True,
#     )
# text = tokenizer.decode(outputs["sequences"][0]).strip()
# print(text)
