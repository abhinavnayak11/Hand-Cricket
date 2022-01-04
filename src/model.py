from torch import nn
import torchvision

def get_model():

    model = torchvision.models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 500),
        # when using CrossEntropyLoss(), the output from NN should be logit values for each class
        # nn.Linear(500,1)
        # when using BCEWithLogitsLoss(), the output from NN should be logit value for True label
        nn.Linear(500, 7)
    )
    return model