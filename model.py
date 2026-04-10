import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import Trainer, TrainingArguments
from typing import List
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import time


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class VLMConfig(PretrainedConfig):
    model_type = 'qwen-vit'
    def __init__(
            self,
            ve_dim:int = 768,
            lm_dim:int = 896,
            image_pad_num:int = 49,
            vit_path='./model/clip-vit-base-patch16',
            llm_path='./model/qwen2.5-0.5b-instruct',
            image_pad = '<|image_pad|>',
            freeze_vit=True,
            freeze_llm=True,
            **kwargs
            ):
        self.ve_dim = ve_dim
        self.lm_dim = lm_dim
        self.image_pad_num = image_pad_num
        self.vit_path = vit_path
        self.llm_path = llm_path
        self.image_pad = image_pad
        self.freeze_vit = freeze_vit
        self.freeze_llm = freeze_llm
        super().__init__(**kwargs)



class VisionProjector(nn.Module):
    def __init__(self, ve_dim: int = 768, lm_dim: int = 896):
        super().__init__()
        self.ve_dim = ve_dim
        self.lm_dim = lm_dim
        self.linear1 = nn.Linear(self.ve_dim * 4, self.lm_dim)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(self.lm_dim, self.lm_dim)

    def forward(self, vision_encoder):
        vision_proj = self.linear1(vision_encoder)
        vision_proj = self.silu(vision_proj)
        return self.linear2(vision_proj)



class VitModel(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config:VLMConfig=None):

        if not config: config = VLMConfig()

        super().__init__(config)
        self.config = config
        self.text_model, self.tokenizer = self.__class__.get_text_model(self.config.llm_path, self.config.freeze_llm)
        self.image_model, self.processor = self.__class__.get_image_model(self.config.vit_path, self.config.freeze_vit)
        self.vision_proj = VisionProjector(self.image_model.config.vision_config.hidden_size, self.text_model.config.hidden_size)

    @staticmethod
    def get_text_model(llm_path = './model/qwen2.5-0.5b-instruct', freeze_llm=True):
        text_model = AutoModelForCausalLM.from_pretrained(llm_path).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(llm_path)

        if freeze_llm:
            for param in text_model.parameters():
                param.requires_grad = False

        return text_model, tokenizer
    
    @staticmethod
    def get_image_model(vit_path = './model/siglip-base-patch16-224', freeze_vit=True):
        image_model = AutoModel.from_pretrained(vit_path).to('cuda')
        processor = AutoProcessor.from_pretrained(vit_path)
        if freeze_vit:
            for param in image_model.parameters():
                param.requires_grad = False
        return image_model, processor
    

    def forward(self, input_ids, pixel_values, labels, attention_mask=None):
        input_ids = input_ids.to(self.text_model.device)
        text_embeds = self.text_model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            pixel_values = pixel_values.to(self.image_model.device)
            image_embeds = self.image_model.vision_model(pixel_values).last_hidden_state
            batch, seq_len, hidden_size = image_embeds.shape
            image_embeds = image_embeds.view(batch, -1, hidden_size*4)
            image_feature = self.vision_proj(image_embeds)
            image_feature = image_feature.to(text_embeds.dtype)
            input_embeds = self.merge_inputs(image_feature, text_embeds, input_ids)
        else:
            input_embeds = text_embeds
        outputs = self.text_model(inputs_embeds=input_embeds, attention_mask=attention_mask)
        logits = outputs.logits
    
        if labels is not None:
            labels = labels.to(self.text_model.device)
            loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

            loss = loss_fn(
                logits.view(-1, logits.shape[-1]), labels.view(-1).to(logits.device)
                )
    
        else:
            loss = None
        return CausalLMOutputWithPast(loss=loss, logits=logits)


    def merge_inputs(self, image_feature, text_embeds, input_ids):

        batch, seq_len, dim = image_feature.shape
        image_pad_ids = self.tokenizer(self.config.image_pad, return_tensors='pt')['input_ids'][0].to(input_ids.device)   
        ufbatch_indices, ufimage_indices = torch.where(input_ids == image_pad_ids)

        def find_indices(ufbatch_indices, ufimage_indices, pad_num):
            bat_indices = []
            img_indices = []

            if len(ufbatch_indices) > pad_num:
                for idx in range(len(ufbatch_indices)):
                    if idx % pad_num == 0:
                        bat_indices.append(ufbatch_indices[idx])

            else:
                bat_indices.append(ufbatch_indices[0])

            if len(ufimage_indices) > pad_num:
                for idx in range(len(ufimage_indices)):
                    if idx % pad_num == 0:
                        img_indices.append(ufimage_indices[idx])
            else:
                img_indices.append(ufimage_indices[0])

            if len(bat_indices) > 1:
                bat_indices = [x.unsqueeze(0) for x in bat_indices]
                bat_indices = torch.cat(bat_indices, dim=0)
            else:
                bat_indices = bat_indices[0]
                bat_indices = bat_indices.unsqueeze(0)

            if len(img_indices) > 1:
                img_indices = [x.unsqueeze(0) for x in img_indices]
                img_indices = torch.cat(img_indices, dim=0)
            else:
                img_indices = img_indices[0]
                img_indices = img_indices.unsqueeze(0)

            return bat_indices, img_indices

        batch_indices, image_indices = find_indices(ufbatch_indices, ufimage_indices, self.config.image_pad_num)
        new_embeds_list = []

        for batch_idx, pad_idx in zip(batch_indices, image_indices):
            text_before = text_embeds[batch_idx:batch_idx + 1, :pad_idx, :]
            text_after = text_embeds[batch_idx:batch_idx + 1, pad_idx + self.config.image_pad_num:, :]
            image_feature_slice = image_feature[batch_idx:batch_idx + 1, :, :]

            assert text_before.shape[0] == image_feature_slice.shape[0], "Batch_dims are not equal"
            assert text_before.shape[2] == image_feature_slice.shape[2], "Hidden_dims are not equal"
            new_embeds_list.append(torch.cat([text_before, image_feature_slice, text_after], dim=1))

        if len(new_embeds_list) > 1:
            input_embeds = torch.cat(new_embeds_list, dim=0)
        else:
            input_embeds = new_embeds_list[0]

        return input_embeds