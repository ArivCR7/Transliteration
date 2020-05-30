from flask import Flask,request,send_file
from flask_cors import CORS
from PIL import Image
import os
import text_detector

app = Flask(__name__)
savePath = "./testdir/"

@app.route("/segment-text")
def search():
    #print(request)
    image_file = request.files['image']
    image = Image.open(image_file)
    filename = image_file.filename
    image.save(savePath + filename, quality=100)  
    outPath = text_detector.detect_text(gpu_list=0, output_dir='./outdir/', checkpoint_path='./checkpoints/')
    return send_file(outPath, mimetype='image/jpg')

if __name__ == '__main__':
    app.run(host="localhost",port=5000,threaded=True,debug=False)
