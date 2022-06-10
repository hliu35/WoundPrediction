import torch
import torch.nn as nn
import copy
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Encoder model Pytorch. 
    """   
    def __init__(self):
        # Initialize self._modules as OrderedDict
        super(Encoder, self).__init__() 
        # Initialize densenet121
        self.embed_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
        # Remove Classifying layer
        self.embed_model = nn.Sequential(*list(self.embed_model.children())[:-1])
        # 7x7 average pool layer
        self.avg = nn.AvgPool2d(kernel_size=7, stride=1)
        # Left image connected layers
        self.fc_16 = nn.Linear(1024, 16)


    def forward(self, x):
        # Embed Left
        u1 = self.embed_model(x)
        u1 = self.avg(u1)
        u1 = u1.view(-1,1024)
        u1 = self.fc_16(u1)
        #u1 = torch.relu(u1)
        return u1

    def load_embed_wts(self, device):
        """
        load pretrained model weights, use only when transfer learning from ImageNET data
        """
        # Initialize densenet121
        self.embed_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True).to(device)

        # Remove Classifying layer
        self.embed_model = nn.Sequential(*list(self.embed_model.children())[:-1])


class HealNet(nn.Module):
    """
    HealNet model Pytorch. 
    """
    def __init__(self):
        # Initialize self._modules as OrderedDict
        super(HealNet, self).__init__() 
        self.encoder = Encoder()
        self.fc_classify = nn.Linear(32, 2)

    def load_encoder_wts(self,device):
        """
        load pretrained model weights, use only when transfer learning from ImageNET data
        """
        # Initialize densenet121
        self.encoder.load_embed_wts(device)


    def forward(self, x):
        """
        forward call to Healnet
        Args:
            x: a tensorized tuple of mouse images of size BSx2x3x224x224
                in the form Batch size by (left image, right image).
        Return:
            preds: logits for BCE.
        """

        # Embed Left
        u1 = self.encoder(x[:,0,:,:,:])


        # Embed Reft
        u2 = self.encoder(x[:,1,:,:,:])

        #Return logits
        return self.fc_classify(torch.cat((u1,u2),1))



class Classifier_Encoder(nn.Module):
    """
    Encoder model Pytorch. 
    """   
    def __init__(self):
        num_classes = 4
        # Initialize self._modules as OrderedDict
        super(Classifier_Encoder, self).__init__() 
        # Initialize densenet121
        #self.embed_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
        #self.embed_model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)

        healnet = HealNet()
        path_to_healnet = "../checkpoints/normalized_healnet_10epoch.tar" # Path to healnet weights
        healnet.load_state_dict(torch.load(path_to_healnet))
        self.embed_model = healnet.encoder

        self.dense = nn.Linear(16, 8)
        self.classifier= nn.Linear(8, num_classes)
        # adding an alternative layer for AB test
        self.direct_classifier = nn.Linear(16, num_classes)

        self.softmax = nn.Softmax()

    def forward(self, x):
        u1 = self.embed_model(x)
        embeddings = u1
        # u1 = torch.relu(self.dense(u1))
        # u1 = self.classifier(u1)
        u1 = self.direct_classifier(u1)
        u1 = self.softmax(u1)  # I'm uncommenting this
        return u1, embeddings

    # def load_weights(self, weight_dict):
    #     """
    #     Load Weights from the trained healnet.encoder upto fc_16
    #     Args:
    #         weight_dict: Pytorch encoder model state_dict()
    #     """
    #     for k,v in zip( weight_dict.keys(), self.state_dict().keys()):
    #         if k in  self.state_dict().keys():
    #           self.state_dict()[v] = copy.deepcopy(weight_dict[k])
