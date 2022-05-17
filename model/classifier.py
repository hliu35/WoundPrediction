import torch
import torch.nn as nn
import copy
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Classifier_Encoder(nn.Module):
    """
    Encoder model Pytorch. 
    """   
    def __init__(self):
        num_classes = 4
        # Initialize self._modules as OrderedDict
        super(Classifier_Encoder, self).__init__() 
        # Initialize densenet121
        self.embed_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
        # Remove Classifying layer
        self.embed_model = nn.Sequential(*list(self.embed_model.children())[:-1])
        # 7x7 average pool layer
        self.avg = nn.AvgPool2d(kernel_size=7, stride=1)
        # Left image connected layers
        self.fc_16 = nn.Linear(1024, 16)
        self.dense = nn.Linear(16, 8)
        self.classifier= nn.Linear(8, num_classes)
        #self.softmax = nn.Softmax()

    def forward(self, x):
        u1 = self.embed_model(x)
        u1 = self.avg(u1)
        u1 = u1.view(-1,1024)
        u1 = self.fc_16(u1)
        embeddings = u1
        u1 = self.dense(u1)
        u1 = torch.relu(u1)
        u1 = self.classifier(u1)
        #u1 = self.softmax(u1)
        return u1, embeddings

    def load_weights(self, weight_dict):
        """
        Load Weights from the trained healnet.encoder upto fc_16
        Args:
            weight_dict: Pytorch encoder model state_dict()
        """
        for k,v in zip( weight_dict.keys(), self.state_dict().keys()):
            if k in  self.state_dict().keys():
              self.state_dict()[v] = copy.deepcopy(weight_dict[k])


def loadClassifier():
    classifier_encoder_test = Classifier_Encoder()
    classifier_encoder_test.to(device)
    path_to_class = "model/best_classifier.tar" # Path to classifier weights
    classifier_encoder_test.load_state_dict(torch.load(path_to_class))
    return classifier_encoder_test


#Test case of how to use classifier
classifier = loadClassifier()
imgPath = "data_cropped\Day 0_A8-1-L.png"
with open(imgPath, 'rb') as f:
    img = Image.open(f)
    img.convert('RGB')

#Transforms used to turn the img into 244,244,3 that densenet needs and into a Tensor
#Normalize as well
transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
img_tensor = transform(img)

#Format into torch stack to be used in classfier
img_tensor = torch.stack([img_tensor])

#set classifier to eval
classifier.eval()
#Classifier returns the final predictions and 16 embeddings saved respectively
labelpred, embeddings = classifier(img_tensor.to(device))

print(F.softmax(labelpred), embeddings)


