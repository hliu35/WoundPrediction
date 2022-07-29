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
finLab = []
wait = 0
count = 0
imgPaths = []
for imgPath in os.listdir('images'):
    imgPaths.append(imgPath)

imgPaths.sort()
for imgPath in imgPaths:
    with open('images/' + imgPath, 'rb') as f:
        img = Image.open('images/' + imgPath)
        img.convert('RGB')
    img_tensor = transform(img)
    img_tensor = torch.stack([img_tensor]).to(device)
    labelpred, _ = Classifier(img_tensor)
    labels.append(np.array2string(F.softmax(labelpred).cpu().detach().numpy(), separator=', '))
    finLab.append(np.array2string((torch.argmax(labelpred) + 1).cpu().detach().numpy(), separator=', '))
    del labelpred, img_tensor
    torch.cuda.empty_cache()

with open('dist.csv', mode = 'w', newline="") as f:
    writer = csv.writer(f)
    for lab in labels:
        writer.writerow([lab])
with open('labels.csv', mode = 'w', newline="") as f:
    writer = csv.writer(f)
    for lab in finLab:
        writer.writerow([lab])

print(len(labels))