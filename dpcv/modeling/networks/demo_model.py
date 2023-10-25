import torch
import torchvision.models as models
from dpcv.modeling.networks.build import NETWORK_REGISTRY


@NETWORK_REGISTRY.register()
def demo_model(args=None):
    model = models.vgg16(num_classes=5)
    model = model.to(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    return model