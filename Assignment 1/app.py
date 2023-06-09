import gradio as gr
#import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

#num_captions = gr.Dropdown([1, 2, 3, 4,5],label = "select no.of captions to generate")

def caption_generator(image, num_captions):
     num_captions = int(float(num_captions))
     raw_image = Image.fromarray(image).convert('RGB')
     inputs = processor(raw_image, return_tensors="pt")
     out = model.generate(
            **inputs,
            num_return_sequences=num_captions, # generate 3 captions
            max_length=32, # maximum length of generated captions
            early_stopping=True, # stop generating captions when all beam hypotheses have finished
            num_beams=num_captions, # number of beams for beam search
            no_repeat_ngram_size=2, # avoid repeating n-grams of size 2 or larger
            length_penalty=0.8 # higher penalty value will encourage shorter captions
             )
     captions = ""
     for i, caption in enumerate(out):
            captions = captions  +processor.decode(caption, skip_special_tokens=True) + " ,"
     return captions 

gr.Interface(caption_generator, inputs= [gr.inputs.Image(), gr.Dropdown([1, 2, 3, 4,5],label = "select no.of captions to generate")], outputs = gr.outputs.Textbox(), live = True).launch()