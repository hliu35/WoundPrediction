import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
print(model.eval())