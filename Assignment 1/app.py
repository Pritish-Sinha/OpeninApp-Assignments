import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_generator(image, num_captions):
    num_captions = int(float(num_captions))
    raw_image = Image.fromarray(image).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(
        **inputs,
        num_return_sequences=num_captions,
        max_length=32,
        early_stopping=True,
        num_beams=num_captions,
        no_repeat_ngram_size=2,
        length_penalty=0.8
    )
    captions = ""
    for i, caption in enumerate(out):
        captions += processor.decode(caption, skip_special_tokens=True) + " ,"
    return captions 

def photo_upload(photo, num_captions):
    captions = caption_generator(photo, num_captions)
    return captions

# Define the inputs and outputs for Gradio interface
photo_input = gr.inputs.Image(label="Upload Photo")
num_captions_input = gr.inputs.Dropdown([1, 2, 3, 4, 5], label="Select number of captions to generate")
caption_output = gr.outputs.Textbox(label="Captions")

# Create the Gradio interface
interface = gr.Interface(fn=photo_upload, inputs=[photo_input, num_captions_input], outputs=caption_output, 
                        title="AI Image Captioning", sidebar=True)

# Launch the interface
interface.launch()
