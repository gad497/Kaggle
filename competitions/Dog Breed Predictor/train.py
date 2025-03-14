import torch
import pandas as pd
import os
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary

class CustomImageDataset(Dataset):
    def __init__(self,img_dir,csv_file=None,transform=None):
        self.img_dir = img_dir
        self.images = os.listdir(img_dir)
        self.csv_file = csv_file
        if self.csv_file:
            self.annotations = pd.read_csv(csv_file)
            self.label_encoder = LabelEncoder()
            self.annotations["breed"] = self.label_encoder.fit_transform(self.annotations["breed"])
        self.transform = transform
    def __len__(self):
        return len(self.annotations) if self.csv_file else len(self.images)
    def __getitem__(self,idx):
        img_name = os.path.join(self.img_dir, self.annotations.iloc[idx,0] + ".jpg")
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.csv_file:
            label = int(self.annotations.iloc[idx,1])
            return image,label
        else:
            return image
        
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class CustomResNet(nn.Module):
    def __init__(self,num_classes):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.resnet.fc.in_features
        # self.resnet.fc = nn.Identity()
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # summary(model=self.resnet,
        #     input_size=(32,3,224,224),
        #     col_names=["input_size","output_size","num_params","trainable"],
        #     col_width=20,
        #     row_settings=["var_names"]
        # )
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features,4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096,2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024,num_classes)
        )
        # summary(model=self.resnet,
        #     input_size=(32,3,224,224),
        #     col_names=["input_size","output_size","num_params","trainable"],
        #     col_width=20,
        #     row_settings=["var_names"]
        # )
    def forward(self,x):
        features = self.resnet(x)
        x = self.fc(features)
        return x
    
# model = CustomResNet(num_classes)
# print(model)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


if __name__ == "__main__":
    train_img_dir = "Data\\train"
    test_img_dir = "Data\\test"
    csv_file_path = "Data\\labels.csv"
    train_dataset = CustomImageDataset(img_dir=train_img_dir,csv_file=csv_file_path,transform=transform)
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(0.1*dataset_size)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(train_dataset,batch_size=32,sampler=train_sampler)
    val_loader = DataLoader(train_dataset,batch_size=32,sampler=val_sampler)
    test_dataset = CustomImageDataset(img_dir=test_img_dir, transform=transform)
    test_loader = DataLoader(test_dataset,batch_size=32,shuffle=True)
    # image, labels = next(iter(train_loader))
    # plt.imshow(image[0].permute(1, 2, 0))
    # plt.show()
    df = pd.read_csv(csv_file_path)
    num_classes = df["breed"].nunique()
    # print(num_classes)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = CustomResNet(num_classes).to(device)
    summary(model=net,
            input_size=(32,3,224,224),
            col_names=["input_size","output_size","num_params","trainable"],
            col_width=20,
            row_settings=["var_names"]
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.fc.parameters(),lr=0.0001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)


    early_stopping = EarlyStopper(patience=3, min_delta=0.01)

    NUM_EPOCHS = 30
    print(f"Training on device {device} for {NUM_EPOCHS} epochs")
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        running_acc = 0.0
        net.train()
        for i, (images, labels) in enumerate(train_loader):
            print(f"training on batch {i+1}",end=", ")
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pred_class = torch.argmax(outputs, dim=1)
            running_acc += (pred_class == labels).sum().item()/len(outputs)
            print(f"running loss : {running_loss:.4f} ",end="\r",flush=True)
        print(f"Epoch {epoch+1}, loss: {running_loss/len(train_loader):.4f}, acc: {(running_acc/len(train_loader))*100:.2f}%",end=", ")
        # scheduler.step()
        net.eval()
        val_loss, val_acc = 0,0
        with torch.no_grad():
            for images,labels in val_loader:
                images,labels = images.to(device),labels.to(device)
                val_outputs = net(images)
                loss = loss_fn(val_outputs,labels)
                val_loss += loss.item()
                pred_class = torch.argmax(val_outputs, dim=1)
                val_acc += (pred_class == labels).sum().item()/len(val_outputs)
            scheduler.step(val_loss/len(val_loader))
            if early_stopping.early_stop(val_loss/len(val_loader)):
                print("Early stopping triggered!")
                break
            print(f"val_loss: {val_loss/len(val_loader):.4f}, val_acc: {(val_acc/len(val_loader))*100:.2f}%")
    print("Training Complete")
    print("Saving model")
    torch.save(net.state_dict(), "model.pth")
    print("Model saved")
