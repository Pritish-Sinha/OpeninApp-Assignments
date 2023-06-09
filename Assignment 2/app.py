import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

import re 
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

stopwords = nltk.corpus.stopwords.words('english')
wn = nltk.WordNetLemmatizer()

def hashtag_generator(image):
     raw_image = Image.fromarray(image).convert('RGB')
     inputs = processor(raw_image, return_tensors="pt")
     out = model.generate(
            **inputs,
            num_return_sequences=4, 
            max_length=32, 
            early_stopping=True, 
            num_beams=4, 
            no_repeat_ngram_size=2, 
            length_penalty=0.8 
             )
     captions = ""
     for i, caption in enumerate(out):
            captions = captions  +processor.decode(caption, skip_special_tokens=True) + " ,"

     text = "".join([word.lower() for word in captions if word not in string.punctuation])
     tokens = re.split('\W+', text)
     text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
     words = set(text)
     hashtags = ""
     for hashtag in words:
         if len(hashtag) == 0:
             pass
         else:
             hashtags = hashtags + f" ,#{hashtag}"
     return hashtags[2:]

gr.Interface(hashtag_generator, inputs= gr.inputs.Image(), outputs = gr.outputs.Textbox(), live = True).launch()
