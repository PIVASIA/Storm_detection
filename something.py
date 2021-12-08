from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from model import RetinaNetModel
 
# load in the hparams yaml file
hparams = OmegaConf.load("hparams.yaml")

# instantiate lightning module
model = RetinaNetModel(hparams)
 
# Instantiate Trainer
trainer = Trainer()
# start train
trainer.fit(model)
# to test model using COCO API
trainer.test(model)