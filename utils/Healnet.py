import torch
import torch.nn as nn
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

healnet = HealNet()
healnet.to(device)

path_to_healnet = "utils/healnet_pretext_45epoch.tar"
healnet.load_state_dict(torch.load(path_to_healnet))

print(healnet)