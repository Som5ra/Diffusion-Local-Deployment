
from datasets import load_dataset
from PIL import Image  
import diffusers
from diffusers import EulerAncestralDiscreteScheduler
import torch
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp


class StableDiffusionInference:
    def __init__(self,
                 text2img_model_id = "prompthero/openjourney-v4"):
        
        self.txt2img_model = diffusers.StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path = text2img_model_id, 
            torch_dtype = torch.float16,
            use_safetensors = True
        )
        # EulerAncestralDiscreteScheduler can generate high quality results with as little as 30 steps.
        self.txt2img_model.scheduler = EulerAncestralDiscreteScheduler.from_config(self.txt2img_model.scheduler.config)

        # self.img2img_model = diffusers.StableDiffusionImg2ImgPipeline.from_pretrained(
        #     pretrained_model_name_or_path = text2img_model_id, 
        #     torch_dtype = torch.float16,

        #     use_safetensors = True
        # )

        # save RAM to load model by using same components
        self.img2img_model = diffusers.StableDiffusionImg2ImgPipeline(**self.txt2img_model.components)
        # EulerAncestralDiscreteScheduler can generate high quality results with as little as 30 steps.
        self.img2img_model.scheduler = EulerAncestralDiscreteScheduler.from_config(self.img2img_model.scheduler.config)
        
        
        
        
        # self.imginpaint_model = diffusers.StableDiffusionInpaintPipeline.from_pretrained(
        #     pretrained_model_name_or_path = text2img_model_id, 
        #     torch_dtype = torch.float16,
        #     use_safetensors = True
        # )


        # save RAM to load model by using same components
        self.imginpaint_model = diffusers.StableDiffusionInpaintPipeline(**self.txt2img_model.components)
        # EulerAncestralDiscreteScheduler can generate high quality results with as little as 30 steps.
        self.imginpaint_model.scheduler = EulerAncestralDiscreteScheduler.from_config(self.imginpaint_model.scheduler.config)


        self.txt2img_model.to("cuda")
        self.img2img_model.to("cuda")
        self.imginpaint_model.to("cuda")

        # less memory and faster inference
        # self.txt2img_model.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        # self.img2img_model.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        # self.imginpaint_model.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
    
    def txt2img(self, prompt, negative_prompt, height = 1024, width = 1024, guidance_scale = 7.5, num_inference_steps = 50):
        with torch.inference_mode():
            ret = self.txt2img_model(
                prompt = prompt,
                negative_prompt = negative_prompt,
                height = height,
                width = width,
                guidance_scale = guidance_scale,
                num_inference_steps = num_inference_steps
            ).images
            return ret

    def img2img(self, prompt, negative_prompt, images, strength = 0.8, guidance_scale = 7.5, num_inference_steps = 50):
        with torch.inference_mode():
            ret = self.img2img_model(
                prompt = prompt,
                negative_prompt = negative_prompt,
                image=images,
                strength = strength,
                guidance_scale = guidance_scale,
                num_inference_steps = num_inference_steps
            ).images
            return ret
        
    def img_inpaint(self, prompt, negative_prompt, images, mask_images, strength = 1.0, guidance_scale = 7.5, num_inference_steps = 50):
        with torch.inference_mode():
            ret = self.imginpaint_model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=images,
                mask_image=mask_images,
                strength = strength,
                guidance_scale = guidance_scale,
                num_inference_steps = num_inference_steps
            ).images
            return ret
    