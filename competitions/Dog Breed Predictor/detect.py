import torch
from train import CustomResNet
from PIL import Image
import os, random
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CustomResNet(120).to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dir = "Data/test/"
file = random.choice(os.listdir(dir))
path = dir + file
# image = transform(Image.open(path).convert("RGB"))
original_img = Image.open(path)
image = transform(original_img)

output = model(image.unsqueeze(0).to(device))

annotations = pd.read_csv("Data/labels.csv")
label_encoder = LabelEncoder()
label_encoder.fit_transform(annotations["breed"])

index = torch.argmax(output,dim=1).to("cpu")

plt.imshow(original_img)
plt.axis("off")
plt.title(label_encoder.inverse_transform(index)[0])
plt.show()