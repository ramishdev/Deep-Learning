import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np

class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
            
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        #TODO
        self._optim.zero_grad()
        out = self._model(x)
        loss = self._crit(out, y.float())
        loss.backward()
        self._optim.step()
        return loss.item()
        
    
    def val_test_step(self, x, y):
        
        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        #TODO
        out = self._model(x)
        loss = self._crit(out, y.float())
        out = out.detach().cpu().numpy()
        pred_0 = np.array(out[:, 0] > 0.5).astype(int)
        pred_1 = np.array(out[:, 1] > 0.5).astype(int)
        pred = np.stack([pred_0, pred_1], axis=1)
        return loss.item(), pred
        
    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        #TODO
        self._model = self._model.train()
        avg_loss = 0
        for x, y in self._train_dl:
            if self._cuda:
                x = x.cuda()
                y = y.cuda()
            loss = self.train_step(x, y)
            avg_loss += loss / len(self._train_dl)
        return avg_loss
    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore. 
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        #TODO
        self._model = self._model.eval()
        avg_loss = 0
        preditions = []
        labels = []
        with t.no_grad():
            for x, y in self._val_test_dl:
                if self._cuda:
                    x, y = x.cuda(), y.cuda()
                loss, pred = self.val_test_step(x, y)
                avg_loss += loss / len(self._val_test_dl)
                if self._cuda:
                    y = y.cpu() 
                preditions.extend(pred)
                labels.extend(y.numpy())
            preditions, labels = np.array(preditions), np.array(labels)
            score = f1_score(labels, preditions, average='micro')
        return avg_loss, score

    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        #TODO
        train_losses = []
        losses = []
        metrics = []
        epoch_n = 0
        no_improvement_count = 0
        while True:
            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation
            #TODO
            if epoch_n == epochs:
                break
            print('Epoch: %3d'%(epoch_n+1))
            train_loss = self.train_epoch()
            loss, metric = self.val_test()
            # if len(losses)!= 0:
            #     print(loss,min(losses),loss < min(losses))
            if len(losses) != 0 and loss < min(losses):
                no_improvement_count = 0
                self.save_checkpoint(epoch_n+1)
            else:
                no_improvement_count += 1
            train_losses.append(train_loss)
            losses.append(loss)
            metrics.append(metric)
            if self._early_stopping_patience > 0:
                 if no_improvement_count >= self._early_stopping_patience:
                    print(f"Early stopping triggered after {no_improvement_count} epochs without improvement.")
                    break
            epoch_n += 1
            print('\tTrain Loss: %.4f\tVal Loss: %.4f\tVal Metric: %.4f'%(train_loss, loss, metric))
        return train_losses, losses, metrics
                    
        
        
        
