from torchvision import models
from torch import nn


def load_resnet(device="cpu"):
    """
    Returns the Resnet model
    """
    model = models.resnet18(pretrained=True)

    # we don't need the last two layers of classification
    modules = list(model.children())[:-2]
    model = nn.Sequential(*modules)

    for parameter in model.parameters():
        parameter.requires_grad = False  # as we are not training resnet

    if device == "cuda":
        model = model.cuda()
    print(f"Loading Resnet18 with device: {device}.")
    model.eval()

    return model


if __name__ == "__main__":
    model = load_resnet()
    print(model)
