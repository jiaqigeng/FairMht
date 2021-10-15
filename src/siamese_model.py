import torch.nn as nn
import torchvision.models as models
import torch


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.net = nn.Sequential(*list(resnet50.children())[:-2])

    def forward(self, x):
        output = self.net(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )
        self.fc.apply(self.init_weights)

    def forward(self, x1, x2):
        output1 = self.get_embedding(x1)
        output2 = self.get_embedding(x2)
        out = torch.cat((output1, output2), dim=1)
        out = self.fc(out)
        return output1, output2, out

    def get_embedding(self, x):
        out = self.embedding_net(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)
