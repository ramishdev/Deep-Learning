import torch as t
from trainer import Trainer
import sys
import torchvision as tv
import model

epoch = int(sys.argv[1])
#TODO: Enter your model here
res = model.ResNet()

crit = t.nn.BCELoss()
trainer = Trainer(res, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
