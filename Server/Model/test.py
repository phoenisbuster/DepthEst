import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Compose
import networks
from transforms import Resize, NormalizeImage, PrepareForNet, CenterCrop
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_transform = Compose(
    [
        lambda img: {"image": img / 255.0},
        Resize(
            width=384,
            height=384,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
        lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
    ]
)

model = networks.MidasNet()
model.load_state_dict(torch.load("Server/Model/checkpoints/model_ckpt.pt"))

img = cv2.imread("Server/Model/test_img/test2.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_size = img.shape[0] if img.shape[0] <= img.shape[1] else img.shape[1]
img = CenterCrop(input_size)(img)
input_batch = data_transform(img)

with torch.no_grad():
    prediction = model(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

prediction = prediction.cpu().numpy()
byte_im = (prediction/prediction.max()*255)
byte_im = cv2.imencode(".png", byte_im)[1].tobytes()
img = np.frombuffer(byte_im, dtype="uint8")
img = cv2.imdecode(img, 0)
print(byte_im)
plt.imshow(img)
plt.show()