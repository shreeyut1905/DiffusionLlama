import os 
import gradio as gr 
import random 
import torch 
import cv2 
import re
import uuid 
import json 
import pickle 
from PIL import Image,ImageDraw,ImageOps,ImageFont 
import math 
import numpy as np 
import argparse 
import inspect 
import tempfile

from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler, PNDMScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DiffusionPipeline, UniPCMultistepScheduler
from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image


from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool 
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_groq import ChatGroq  

from sentence_transformers import SentenceTransformer
from compel  import Compel,ReturnedEmbeddingsType
from dotenv import load_dotenv
import os
load_dotenv()
PREFIX = """DIffusionLlama is designed to be able to assist user in generating high-quality images. 
Human may provide some text prompts to DIffusionLlama.The input prompts will be analyzed by DIffusionLlama to select most suitable generative models for generating images.
Overall,DIffusionLlama is powerful image generation system that can assist  in procesing various forms of textual input and match them  with the most suitable  generative model to accomplish the generation task.
TOOLS:
-----
DIffusionLlama has access to the following tools:"""
FORMAT_TOOLS = """To Use a Tool please Use a following format:
```
Though: Do I need to use a tool?Yes
Action: the action to take,should be one of [{tool_names}]
Action Input : the input to the action
Observation: the result of the action
```
When you have response to say to Human,or if you do not need to use a tool,you must use the format:
```
Thought:Do I need to use a tool?No
{ai_prefix}:[your response here]
```
"""
SUFFIX = """You are very strict to filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}
New input:{input}
DIffusionLlama must use tools to observe images rather than imagination.
The thoughts and observations are only visible for DiffusionLlama,DiffusionLlama should remember to repeat important information in the final response for Human.
Thought:Do I need to use a tool?{agent_scratchpad} Let's think step by step."""
TOT_PROMPTS = """Identify and behave as five different experts that are appropriate to select one element from the input list that best matches the input prompt.
All experts will write down the selection result, then share it with the group.
You then analyze all 5 analyses and output the consensus selected element or your best guess matched element.
The final selection output must be the same as the TEMPLATE:
TEMPLATE:
```
Selected:[the selected word]
```
Input list:{search_list}

Input prompt:{input}
"""
PROMPT_PARSE_PROMPTS = """Given the user input text.
Please judge the paradigm of the input text,and then recognize the main string of text prompts according to the corresponding form.
The output must be same as  the TEMPLATE:
TEMPLATE:
```
Prompts:[the output prompts]
```
For instance:
1. Input: A dog
   Prompts: A dog 
2. Input: generate an image of a dog
   Prompts: an image of a dog
3. Input: I want to see a beach
   Prompts: a beach
4. Input: If you give me a toy, I will laugh very happily
   Prompts: a toy and a laugh face

Input: {inputs}
"""
TREE_OF_MODEL_PROMPT_SUBJECT = """You are information analyst who can analyze and abstract a set of words to abstract some representation categories.
Below is a template that can respresent the abstracted categories in Subject Dimension belonging to concrete noun:
TEMPLATE:
```
Categories :
-[Subject]
-[Subject]
-...
```
You MUST abstract the categories in a highly abstract manner only from Subject Dimension and ensure the whole number of Categories are fewer than 5.
Then ,  You MUST remove the style-related categories.
Please output the categories following the format of TEMPLATE.
Input : {input}
"""

TREE_OF_MODEL_PROMPT_STYLE = """You are an information analyst who can analyze and summarize a set of words to abstract some representation categories.
Below is a template that can represent the abstracted categories in Style Dimensions:
TEMPLATE:
```
Categories:
-[Style]
-[Style]
-...
```
You MUST abstract the categories in a highly abstract manner from only style dimension and ensure the whole number of categories are fewer than 8.
Please output the Categories following the format of the TEMPLATE.

Input:{input}
"""
TREE_OF_MODEL_PROMPT_ = """You are an information analyst who can create a Knowledge Tree according to the input categories.
Below is a knowledge Tree Template:
TEMPLATE:
```
Knowledge Tree:
-[Subject]
    -[Style]
    - ...
-[Subject]
    -...
```
You MUST place the each style category as subcategory under the subject categories based on whether it can be well matched with a specific subject category to form a reasonable scene.
Please output the categories following the format of TEMPLATE.
Subject Input:{subject}
Style Input:{style}
"""
TREE_OF_MODEL_PROMPT_ADD_MODELS = """You are an information analyst who can add some input models to an input knowledge tree according to similarity of the model tags and the categories of the knowledge tree.
You need to place each input model into the appropriate subcategory on the tree,one by one.
You must  keep the original content of the knowledge tree.
Please output the final knowledge tree.
knowledge Tree Input:{tree}
Models Input:{models}
Model Tags Input:{model_tags}
"""
os.makedirs('image',exist_ok=True)

from langchain.llms.base import LLM

from langchain import PromptTemplate, HuggingFaceHub
from langchain.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed 

def prompts(name,description):
    def decorator(func):
        func.name = name
        func.description = description
        return func
    return decorator
def cut_dialogue_history(history_memory,keep_last_n_words=500):
    if history_memory is None or len(history_memory)==0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)

    if n_tokens < keep_last_n_words:
        return history_memory 
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens  >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)

class Text2Image:
    def __init__(self,device):
        print("Initialising Text2Img to {device}")
        self.device = device 
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.llm = ChatGroq(model = "llama-3.3-70b-versatile",temperature=0,groq_api_key=os.getenv("GROQ_API_KEY"))
        if not os.path.exists('model_tree_tot_sdxl.json'):
            with open('model_data_sdxl.json','r') as f:
                self.model_data_all = json.load(f)
            model_tags = {model["model_name"]:model["tag"] for model in self.model_data_all}
            model_tree = self.build_tree(model_tags)
            model_all_data = {model["model_name"].split(".")[0]:model for model in self.model_data_all}
            save_model_tree = {}
            for cate_name,sub_category in model_tree.items():
                cate_name = cate_name.lower()
                temp_category  = {}

                if "Universal" not in sub_category:
                    temp_category["Universal"] = [model_all_data['kandinsky'],model_all_data["sd_xl"]]
                for sec_cate_name,sub_sub_cates in sub_category.items():
                    sec_cate_name = sec_cate_name.lower()
                    temp_model_list = []

                    for model_name in sub_sub_cates:
                        model_name = model_name.strip()
                        lower_name = model_name[0].lower() + model_name[1:]
                        if model_name in model_all_data:
                            temp_model_list.append(model_all_data[model_name])
                        elif lower_name in model_all_data:
                            temp_model_list.append(model_all_data[lower_name])
                    temp_category[sec_cate_name] = temp_model_list
                save_model_tree[cate_name] =  temp_category
            json_data = json.dumps(save_model_tree,indent=2)
            with open('model_tree_tot_sdxl.json','w') as f:
                f.write(json_data)
                f.close()
        with open('model_tree_tot_sdxl.json','r') as f:
            self.model_all_data = json.load(f)
            self.model_all_data = {model["model_name"]:model for model in self.model_all_data}
        with open('./VectorDB_HF/prompt_embed_st.pickle', 'rb') as f:
            self.pt_pairs = pickle.load(f)
        with open('./VectorDB_HF/prompt2scores_sdxl.json', 'r') as f:
            self.prompt2scores = json.load(f)
        self.st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    def build_tree(self,model_tags):
        tags_only = list(model_tags.values())
        model_names = list(model_tags.keys())
        prompts = TREE_OF_MODEL_PROMPT.format(input=tags_only)
        
                


         
class CoversationBot:
    def __init__(self,load_dict):
        return 