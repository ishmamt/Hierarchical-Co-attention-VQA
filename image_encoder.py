from torchvision import models
from torch import nn
from torch.cuda import get_device_name, current_device


def load_resnet(device):
    """ Returns the ResNet model.
    Parameters:
        device: string; Is the device using CUDA or not.
    Returns:
        model: torch model object; ResNet model.
    """

    model = models.resnet18(pretrained=True)

    # We don't need the last two layers of classification
    modules = list(model.children())[:-2]
    model = nn.Sequential(*modules)

    for parameter in model.parameters():
        parameter.requires_grad = False  # As we are not training

    if device == "cuda":
        model = model.cuda()
        print(f"Loading Resnet18 with: {device} | {get_device_name(current_device())}.")
    else: 
        print(f"Loading Resnet18 with: {device}.")
    model.eval()

    return model


if __name__ == "__main__":
    model = load_resnet("cpu")
    print(model)
