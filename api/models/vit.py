import torch
import torchvision.models
import PIL


class ViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        self.model = torchvision.models.vit_b_16(weights=self.weights)
        self.preprocessing = self.weights.transforms()
        self.h = None
        self._register_hooks()

    def _register_hooks(self):
        self.model.encoder.layers[-1].register_forward_hook(self._hook_fn)

    def _hook_fn(self, m_layer, inpts, outputs):
        self.h = outputs[:, 1:, :]

    def forward(self, x):
        prd = self.model(x)
        return self.h
