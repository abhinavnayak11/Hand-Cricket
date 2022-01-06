from torch import nn
import torchvision

def get_model():

    model = torchvision.models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 500),

        # nn.Linear(500,1) : To be used for binary classification and BCEWithLogitsLoss() 
        # for BCEWithLogitsLoss(), the output from NN should be the logit value for True label

        # nn.Linear(500,2) : To be used for binary classification and CrossEntropyLoss() 
        # for CrossEntropyLoss(), the output from NN should be the logit values for each class

        nn.Linear(500, 7) # To be used for multi-class classification. 
        # when using CrossEntropyLoss(), the output from NN should be logit values for each class
    )
    return model