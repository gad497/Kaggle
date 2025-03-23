import torch
from train import CustomResNet
from PIL import Image
import os, random
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
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

annotations = pd.read_csv("Data/labels.csv")
label_encoder = LabelEncoder()
label_encoder.fit_transform(annotations["breed"])

train_dir = "Data/train/"
image_list = os.listdir(train_dir)[:800]
y_test = []
predictions = []

for file in image_list:
    print(f"predicting for {file}", end="\r", flush=True)
    path = train_dir + file
    id = file.split(".")[0]
    y_test.append(annotations.loc[annotations["id"]==id,"breed"].values[0])
    image = transform(Image.open(path))
    prediction = model(image.unsqueeze(0).to(device))
    index = torch.argmax(prediction,dim=1).to("cpu")
    predictions.append(label_encoder.inverse_transform(index)[0])

cm = confusion_matrix(y_test, predictions)
fig, ax = plt.subplots(figsize=(30,30))
disp = ConfusionMatrixDisplay(cm,display_labels=label_encoder.classes_)
disp.plot(cmap="Blues", ax=ax)  # Pass the same axis

ax.set_xticklabels(label_encoder.classes_,rotation=90)
ax.set_title("Confusion Matrix", fontsize=18)
ax.set_xlabel("Predicted Labels", fontsize=12)
ax.set_ylabel("Actual Labels", fontsize=12)
ax.tick_params(axis='both', labelsize=10)
fig.savefig("confusion_matrix.pdf")
fig.savefig("confusion_matrix.png", dpi=600)

report = classification_report(y_test, predictions, labels=label_encoder.classes_)
fig, ax = plt.subplots(figsize=(30, 30))
ax.text(0.1, 0.9, report, fontsize=12, family='monospace', verticalalignment='top')
ax.axis('off')
plt.savefig("classification_report.pdf")
plt.savefig("classification_report.png", dpi=600)