import torch.nn as nn
import torchvision.models as models


def set_parameter_requires_grad(model, 
                                feature_extracting: bool = False):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name: str, 
                     num_classes: int, 
                     feature_extract: bool = False, 
                     use_pretrained: bool = True):
    input_size = 0

    model = getattr(models, model_name, lambda: None)
    model_ft = model(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)

    if model_name.startswith("resnet"):
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name.startswith("vgg"):
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name.startswith("squeezenet"):
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name.startswith("densenet"):
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name.startswith("inception"):
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        raise ValueError("{0} is not supported!".format(model_name))

    return model_ft, input_size