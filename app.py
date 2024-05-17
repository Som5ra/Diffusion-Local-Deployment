from flask import Flask, render_template, request, url_for, redirect
from inference import StableDiffusionInference
from io import BytesIO
from PIL import Image
import os
import base64

ROOT = os.path.join(os.getcwd(), 'templates')

app = Flask(__name__, template_folder=ROOT)

# sdinference = StableDiffusionInference()

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/upload_image', methods=['POST'])
def upload():
    print("!23456")

    if request.method == 'POST':
        image = request.files['in_image']  # get file
        # image_b64 = base64.b64encode(image.read()).decode('utf-8')
        print(image)

@app.route("/api/txt2img", methods = ["POST"])
def pipeline_txt2img():
    request_json = dict(request.get_json())
    print(request_json)
    prompt = request_json.get("prompt", "")
    negative_prompt = request_json.get("negative_prompt", "")
    width = int(request_json.get("width", 640))
    height = int(request_json.get("height", 640))
    print("prompt: ", prompt)
    print("negative_prompt: ", negative_prompt)
    print("width: ", width)
    print("height: ", height)
    # images = sdinference.txt2img(
    #     prompt=prompt,
    #     negative_prompt = negative_prompt,
    #     width = width,
    #     height=height
    # )
    ret = {
        "images": []
    }
    images = [Image.open("/media/risksis/HDD_1/LECAS/brake_pad/burnt_mark/positive-emsd (test-use)/20220422_103250.jpg")]
    for img in images:
        im_file = BytesIO()
        img.save(im_file, format="PNG")
        im_bytes = im_file.getvalue()
        im_b64 = base64.b64encode(im_bytes)
        image_string = im_b64.decode('ascii')
        ret["images"].append(image_string)

    return ret



if __name__ == '__main__':
    app.run(debug=True)