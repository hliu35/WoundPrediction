from msilib.schema import Class
import temporal_encoder
import os
from PIL import Image
import torchvision.transforms as transforms
import torch 
import torch.nn.functional as F
import numpy as np
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Classifier = temporal_encoder.Classifier_Encoder()
Classifier = Classifier.to(device)
path = "checkpoints/normalized_classifier.tar"
Classifier.load_state_dict(torch.load(path))
Classifier.eval()

transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.56014212, 0.40342121, 0.32133712], [0.20345279, 0.14542403, 0.12238597])])

labels = []
wait = 0 + 55 + 55 + 55 + 55 + 55 + 55 + 55 + 55 + 55 + 55
count = 0
imgPaths = []
for imgPath in os.listdir('images'):
    if(count < wait):
        count += 1
        continue
    if(len(imgPaths) == 55):
        break
    imgPaths.append(imgPath)
for imgPath in imgPaths:
    with open('images/' + imgPath, 'rb') as f:
        img = Image.open('images/' + imgPath)
        img.convert('RGB')
    img_tensor = transform(img)
    img_tensor = torch.stack([img_tensor])
    labelpred, _ = Classifier(img_tensor.to(device))
    labels.append(F.softmax(labelpred))

with open('dist.csv', mode = 'w', newline="") as f:
    writer = csv.writer(f)
    for lab in labels:
        #lab = torch.argmax(lab) + 1
        temp = np.array2string(lab.cpu().detach().numpy(), separator=', ')
        writer.writerow([temp])
with open('labels.csv', mode = 'w', newline="") as f:
    writer = csv.writer(f)
    for lab in labels:
        lab = torch.argmax(lab) + 1
        temp = np.array2string(lab.cpu().detach().numpy(), separator=', ')
        writer.writerow([temp])

print(len(labels))