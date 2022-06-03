from ctypes.wintypes import RGB
from temporal_encoder import loadClassifier
import os
from PIL import Image
import torchvision.transforms as transforms
import torch 
import torch.nn.functional as F
import numpy as np
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Classifier = loadClassifier()
Classifier.eval()

transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

labels = []
count = 0
for imgPath in os.listdir('images'):
    if(count == 55):
        break
    with open('images/' + imgPath, 'rb') as f:
        img = Image.open('images/' + imgPath)
        img.convert('RGB')
    img_tensor = transform(img)
    img_tensor = torch.stack([img_tensor])
    labelpred, _ = Classifier(img_tensor.to(device))
    labels.append(F.softmax(labelpred))
    count += 1

with open('labels.csv', mode = 'w') as f:
    writer = csv.writer(f)
    for lab in labels:
        temp = np.array2string(lab.cpu().detach().numpy()[0], separator=', ')
        writer.writerow([temp])

print(len(labels))