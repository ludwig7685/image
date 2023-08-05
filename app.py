from flask import Flask, render_template, request, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def generate_caption(image):
    raw_image = Image.open(image).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        captions = []
        images = request.files.getlist('image')
        for image in images:
            caption = generate_caption(image)
            captions.append(caption)
        return render_template('index.html', captions=captions)
    return render_template('index.html')

@app.route('/api/generate_captions', methods=['POST'])
def generate_captions_api():
    captions = []
    images = request.files.getlist('image')
    for image in images:
        caption = generate_caption(image)
        captions.append(caption)
    return jsonify({'captions': captions})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


