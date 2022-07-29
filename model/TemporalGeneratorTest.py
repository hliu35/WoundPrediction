import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import csv

def LinearInter(EmbOne, EmbTwo, distEmb, distFin):
    inter = []
    for i, j in zip(EmbOne, EmbTwo):
        diff = j - i
        diff = diff/distEmb
        inter.append(j + diff * distFin)
    return inter

def GenPairs(embedingList, MouseID):
    temp = []
    for i in embedingList:
        if MouseID in i[0]:
            temp.append(i)
    for i in temp:
        i[0], _ = i[0].split("_")
        i[0] = int(i[0])
    temp.sort()
    temp.pop(15)
    pairs = []
    for i in range(len(temp)):
        for j in range(i + 1, len(temp)):
            pairs.append([temp[i],temp[j]])
    return pairs



#Create a list of the embeddings
embeddings = []
with open('data/normalized_embeddings.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
      embeddings.append(row)
embeddings.pop(0)


def Convert(string):
    li = list(string.split(","))
    return li

values = []
for embed in embeddings:
  temp = embed[1].strip("[]")
  temp = temp.replace('\n', "")
  temp = Convert(temp)
  temp = [float(i) for i in temp]
  _, day = embed[0].split()
  day, _ = day.split(".")
  temp.insert(0,day)
  values.append(temp)

MouseID = 'A8-5-R'
temp = GenPairs(values, MouseID)


GenModel = torch.load("models\cgan_gen_0610.pth")
GenModel.eval()


for pair in temp:
    for dist in range(1, 15):
        if (pair[1][0] + dist > 15):
            continue
        emb = LinearInter(pair[0][1:], pair[1][1:], pair[1][0] - pair[0][0], dist)
        emb = torch.stack([torch.Tensor(emb).cuda()])
        noise = Variable(torch.randn((1, 100)).cuda())
        img_shape = (3, 128, 128)
        img = GenModel(noise, emb).view(-1, *img_shape)
        save_image(img.data, "images/" + MouseID + "-" + chr(pair[0][0] + 97) + "-" + chr(pair[1][0]+97) +  "-" + chr(pair[1][0]+dist+97) + ".png")

