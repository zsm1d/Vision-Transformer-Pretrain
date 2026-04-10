import gradio as gr
from model import *
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoProcessor, AutoConfig
import torch
from PIL import Image
import os

device = 'cuda'
tokenizer = AutoTokenizer.from_pretrained('./model/qwen2.5-0.5b-instruct')
processor = AutoProcessor.from_pretrained('./model/siglip-base-patch16-224')
AutoConfig.register('qwen-vit', VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VitModel)

pretrained_model = AutoModelForCausalLM.from_pretrained('./save/pretrained')
pretrained_model.to(device)
pretrained_model.eval()



def generate(text_input, image_input=None, max_newtokens=100, image_pad_num=49, temperature: float=7.0, top_k=10):
    if image_input is not None:
        prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': 'You are a helpful assistant.'}, 
                                                {'role': 'user', 'content': f'{text_input}\n<image>'}], 
                                                tokenize=False, 
                                                add_generation_prompt=True).replace('<image>', '<|image_pad|>' * image_pad_num)
        pixel_values = processor(text=None, images=image_input, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(device)
    else:
        prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': 'You are a helpful assistant.'}, 
                                        {'role': 'user', 'content': f'{text_input}'}], 
                                        tokenize=False, 
                                        add_generation_prompt=True)
    input_ids = tokenizer(text=prompt, return_tensors='pt')['input_ids'].to(device)
    eos = tokenizer.eos_token_id
    previous = input_ids.shape[-1]
    while input_ids.shape[-1] < previous + max_newtokens -1:
        if image_input is not None:
            inferences = pretrained_model(input_ids, pixel_values, None)
        else:
            inferences = pretrained_model(input_ids, None, None)
        logits = inferences.logits
        logits = logits[:, -1, :]
        
        for token in set(input_ids.tolist()[0]):
            logits[:, token] /= 1.0
        
        if temperature == 0.0:
            _, next_idx = torch.topk(input=logits, k=1, dim=-1)
        else:
            logits = logits / temperature
            if top_k is not None:
                topk_value, _ = torch.topk(input=logits, k=min(top_k, logits.shape[-1]), dim=1)
                logits[logits < topk_value[:, [-1]]] = -float('Inf')
            
            probs = torch.softmax(logits, dim=1)
            next_idx = torch.multinomial(probs, num_samples=1, replacement=False, generator=None)

        if next_idx == eos:
            break
        input_ids = torch.cat((input_ids, next_idx), dim=1)

    return tokenizer.decode(input_ids[: ,previous:][0])


with gr.Blocks() as demo:
    with gr.Row():
        # upload
        with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Select Image")
        with gr.Column(scale=1):
            text_input = gr.Textbox(label="Input Text")
            text_output = gr.Textbox(label="Output Text")
            generate_button = gr.Button("Generate")
            generate_button.click(generate, inputs=[text_input, image_input], outputs=text_output)

if __name__ == '__main__':
    demo.launch(share=False, server_name="0.0.0.0", server_port=7888)
    
    