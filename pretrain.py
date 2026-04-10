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
from model import VLMConfig, VitModel



class PretrainDataset(Dataset):
    def __init__(self, image_path, data_path, tokenizer, processor, config: VLMConfig=None):
        super().__init__()
        if not config: config = VLMConfig()
        self.config = config
        self.image_path = image_path
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.data = self.load_data()
    
    def __len__(self):
        return len(self.data)
    
    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def _create_prompt(self, conversations):
        prompt = self.tokenizer.apply_chat_template([{'role': 'system', 'content': 'You are a AI assistant.'}, 
                                                        {'role': 'user', 'content': conversations[0]['value']},], 
                                                        tokenize=False, 
                                                        add_generation_prompt=True, 
                                                        return_tensors=None).replace('<image>', '<|image_pad|>' * self.config.image_pad_num)
        return prompt

    def _create_white_prompt(self, white_image):
        pixel_values = self.processor(text=None, images=white_image, return_tensors='pt').pixel_values
        prompt = self.tokenizer.apply_chat_template([{'role': 'system', 'content': 'You are a AI assistant.'}, 
                                                     {'role': 'user', 'content': 'What is the content of the image?\n<image>'},], 
                                                     tokenize=False, 
                                                     add_generation_prompt=True, 
                                                     return_tensors=None).replace('<image>', '<|image_pad|>' * self.config.image_pad_num)
        return pixel_values, prompt


    def __getitem__(self, index):
        try:
            sample = self.data[index]
            image_id = sample['image']
            conversations = sample['conversations']
            prompt = self._create_prompt(conversations)
            response = conversations[1]['value'] + self.tokenizer.eos_token
            prompt_input_ids = self.tokenizer(prompt, return_tensors=None).input_ids
            response_input_ids = self.tokenizer(response, return_tensors=None).input_ids
            input_ids = prompt_input_ids + response_input_ids
            labels = [self.tokenizer.pad_token_id] * len(prompt_input_ids)  + response_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
            image = Image.open(os.path.join(self.image_path, image_id)).convert('RGB')
            pixel_values = self.processor(text=None, images=image, return_tensors='pt').pixel_values
        except:
            white_image = Image.new('RGB', (224, 224), color='white')
            pixel_values, prompt = self._create_white_prompt(white_image)
            response = 'This is a blank white image' + self.tokenizer.eos_token
            prompt_input_ids = self.tokenizer(prompt, return_tensors=None).input_ids
            response_input_ids = self.tokenizer(response, return_tensors=None).input_ids
            input_ids = prompt_input_ids + response_input_ids
            labels = [self.tokenizer.pad_token_id] * len(prompt_input_ids) + response_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
        return {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'labels': labels,
        }

class PretrainDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        max_len = max([len(feature['input_ids']) for feature in features])
        input_ids = []
        pixel_values = []
        labels = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            pixel_values.append(feature['pixel_values'])
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'pixel_values': torch.cat(pixel_values, dim=0),
            'labels': torch.tensor(labels, dtype=torch.long)
        }




if __name__ == '__main__':
    llm_path='./model/qwen2.5-0.5b-instruct'
    vit_path='./model/siglip-base-patch16-224'
    config = VLMConfig(llm_path=llm_path, vit_path=vit_path, image_pad = '<|image_pad|>', freeze_vit=True, freeze_llm=True)
    model = VitModel(config).to('cuda')
    print(model)
    print(f'The parameters of the model are {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    image_path = './dataset/LLaVA-CC3M-Pretrain-595K/images'
    data_path = './dataset/Chinese-LLaVA-Vision-Instructions/chat-translated.json'
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    processor = AutoProcessor.from_pretrained(vit_path)
    output_dir = './save/pretrained'
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=1e-4,
        weight_decay=0,
        fp16=True,
        eval_strategy='no',
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=PretrainDataset(image_path, data_path, tokenizer, processor, config),
        data_collator=PretrainDataCollator(tokenizer),
    )
    # Check if the model is on GPU
    print(f"Model device: {next(model.parameters()).device}")  # The output should be'cuda:0'

    start_time = time.time()
    trainer.train(resume_from_checkpoint=False)
    torch.cuda.empty_cache()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time // 3600:.0f}hours {(total_time % 3600) // 60:.0f}mins {total_time % 60:.2f}seconds")
    trainer.save_model(output_dir)
    trainer.save_state()