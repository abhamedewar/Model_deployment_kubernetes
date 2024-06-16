import flask
from flask import request
from PIL import Image
from model import Network

import torchvision.transforms as T
import torch
import numpy as np

app = flask.Flask(__name__)

@app.route("/image", methods=["POST"])
def handle_request():
    if request.method == "POST":
        imagefile = flask.request.files["image"]
        img = Image.open(imagefile).convert("L")  # load with Pillow
        img = np.array(img)
        transform = T.Compose([T.ToTensor(), T.Resize((28, 28))])
        img = transform(img)
        model = Network()
        model.load_state_dict(
            torch.load("mnist_new.pth", map_location=torch.device("cpu"))
        )
        model.eval()
        results = model(img)
        print(results)
        category = torch.argmax(results)
        print(category)
        category = str(category.item())
        print(category)
        return category


app.run(host="0.0.0.0", port=88, debug=True)
