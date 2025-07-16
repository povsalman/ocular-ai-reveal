import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=5):
        super(VisionTransformer, self).__init__()
        weights = ViT_B_16_Weights.DEFAULT
        self.model = vit_b_16(weights=weights)
        self.model.heads = nn.Linear(in_features=768, out_features=num_classes)

    def forward(self, x):
        return self.model(x)
