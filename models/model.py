import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from helpers import caffemodel2pytorch
# import caffemodel2pytorch
from models.backbone import initialize_model


class POIClassifier(pl.LightningModule):
    def __init__(self,
                #  backbone='resnet34',
                 n_classes: int = 10, 
                 n_epochs: int = 30, 
                 lr: float = 1e-5,
                 weight_decay: float = 0.01,
                 momentum: float = 0.9):
        super().__init__()
        self.save_hyperparameters()
        
        # self.model = initialize_model(backbone, n_classes)
        # prototxt = 'deploy_alexnet_places365.prototxt'
        # model = 'alexnet_places365.caffemodel'
        prototxt ='deploy_vgg16_places365.prototxt'
        caffemodel = 'vgg16_places365.caffemodel'
        # caffe_proto = 'https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto'
        self.model = caffemodel2pytorch.Net(
            prototxt = prototxt,
            weights = caffemodel,
            caffe_proto = 'https://raw.githubusercontent.com/BVLC/caffe/master/src/caffe/proto/caffe.proto'
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.model.cuda()
        
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum

        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        output = self.model(x)
        return output
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), 
                              lr=self.lr, 
                              momentum=self.momentum)
        return optimizer 

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs = self.model(inputs)
        # outputs = self(inputs)
        loss = self.criterion(outputs['prob'], labels)
        
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        outputs = self.model(inputs)
        # outputs = self(inputs)
        loss = self.criterion(outputs['prob'], labels)
        
        self.log('val_loss', loss, prog_bar=True, logger=True)