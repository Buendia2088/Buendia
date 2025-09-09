import torch
from torch import nn
from model import *
import numpy as np
from sklearn.metrics import f1_score
from dataset import MyDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from util import setup_seeds
from torchvision.transforms import Lambda
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


class Trainer:
    '''
    You are supposed to implement the Trainer following the instructions
    '''
    def __init__(
          self,
          data_dir: str = "data",
          log_dir: str = "log",
          exp_name: str = "second_take_MLP_128",    
          model_name: str = "MLP_D",
          epochs: int = 50,
          device: str =  "cpu",
          batch_size: int = 64,
          lr: float = 0.001,
          weight_decay: float = 0.0001,
          random_seed: int = 0,
          optimizer=torch.optim.Adam,
     ):
        setup_seeds(random_seed)
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_losses = [] 
        self.val_losses = []
        self.test_losses = []

        if model_name == 'MLP':
            self.model = MLP(in_channels=3*32*32, hidden_channels=128, out_channels=10)
            transform = Compose([Lambda(lambda x: x.view(-1))])

        elif model_name == 'MLP_D':
            self.model = MLP_D(in_channels=3*32*32, hidden_channels=128, out_channels=10, dropout_rate=0.5)
            transform = Compose([Lambda(lambda x: x.view(-1))])
                
        elif model_name == 'CNN':
            transform = None
            self.model = CNN(in_channels=3, hidden_channels=16, out_channels=10)
        self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.criterion = nn.CrossEntropyLoss()

        train_dataset = MyDataset(data_dir=data_dir, transform=transform, mode='train')
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = MyDataset(data_dir=data_dir, transform=transform, mode='val')
        self.val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = MyDataset(data_dir=data_dir, transform=transform, mode='test')
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        self.log_file = log_dir + "/" + exp_name + ".txt"
        f = open(self.log_file, "w")
        info = f"exp name{exp_name}, model name: {model_name}, epochs: {epochs}, device: {device}, batch size: {batch_size}, lr: {lr}, weight decay: {weight_decay} random seed: {random_seed}\n"
        f.write(info)
        
    def load_checkpoint(self, checkpoint_path: str):
        '''
        load model and optimizer checkpoint
        learn how to save checkpoint by save_checkpoint function and implement load_checkpoint
        '''
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def save_checkpoint(self, checkpoint_path: str):
        '''
        save model and optimizer checkpoint
        '''
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        for x_val, y_val in self.val_dataloader:
            x_val, y_val = x_val.to(self.device), y_val.to(self.device)

            outputs = self.model(x_val)
            loss = self.criterion(outputs, y_val)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            y_true.extend(y_val.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            total += y_val.size(0)
            correct += (predicted == y_val).sum().item()

        val_loss /= len(self.val_dataloader)
        val_accuracy = correct / total
        conf_matrix = confusion_matrix(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')

        log_content = f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1 Score: {f1:.4f}\n'
        log_content += 'Validatoin Confusion Matrix:\n'
        log_content += f'{conf_matrix}\n'
        # log_content += f'Best hyperparameters: batch_size={self.batch_size}, lr={self.lr}, weight_decay={self.weight_decay}\n'
        print(log_content)
        with open(self.log_file, "a") as f:
            f.write(log_content)

        self.val_losses.append(val_loss)

        # return val_accuracy


    @torch.no_grad()
    def test(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []

        for x_test, y_test in self.test_dataloader:
            x_test, y_test = x_test.to(self.device), y_test.to(self.device)
            outputs = self.model(x_test)
            loss = self.criterion(outputs, y_test)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y_test.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()

        test_loss /= len(self.test_dataloader)
        test_accuracy = correct / total
        conf_matrix = confusion_matrix(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        log_content = f'Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1 Score: {f1:.4f}\n'
        log_content += 'Test Confusion Matrix:\n'
        log_content += f'{conf_matrix}\n'
        print(log_content)
        with open(self.log_file, "a") as f:
            f.write(log_content)

    def train(self):
        for epoch in range(self.epochs):
            # why do we need to set self.model.train()
            self.model.train()
            for step, (x, y) in enumerate(self.train_dataloader):
                
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x)

                loss = self.criterion(outputs, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # log every 200 steps
                if step % 200 == 0:
                    print(f"Epoch [{epoch}/{self.epochs}], Step [{step}/{len(self.train_dataloader)}], Train Loss: {loss.item()}")
                    with open(self.log_file, "a") as f:
                        f.write(f"Epoch [{epoch}/{self.epochs}], Step [{step}/{len(self.train_dataloader)}], Train Loss: {loss.item()}\n")
                       
            self.train_losses.append(loss.item())     
            self.validate()

            # think when to test
        self.test()

        with open(self.log_file, "a") as f:
            f.write("Training finished.\n")

        self.save_checkpoint("model_checkpoint.pth")

    def plot_losses(self):
        plt.plot(range(len(self.train_losses)), self.train_losses, label='Train Loss')
        plt.plot(range(len(self.val_losses)), self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('MLP Training and Validation Losses')
        plt.legend()
        plt.savefig('MLP_loss_curve.png')
        plt.show()

def main():  
    model_name = input("Enter model name: (MLP/MLP_D(MLP_DROPOUT)/CNN)")

    ''''
    candidate_batch_sizes = [32, 64, 128]
    candidate_lrs = [0.001, 0.01, 0.1]
    candidate_weight_decays = [0.0001, 0.001, 0.01]

    best_accuracy = 0.0
    best_batch_size = None
    best_lr = None
    best_weight_decay = None

    for batch_size in candidate_batch_sizes:
        for lr in candidate_lrs:
            for weight_decay in candidate_weight_decays:
                print(f"Training with batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}")
                trainer = Trainer(model_name=model_name, batch_size=batch_size, lr=lr, weight_decay=weight_decay)
                trainer.train()
                
                val_accuracy = trainer.validate()

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_batch_size = batch_size
                    best_lr = lr
                    best_weight_decay = weight_decay

    print(f"Best hyperparameters: batch_size={best_batch_size}, lr={best_lr}, weight_decay={best_weight_decay}, Val Accuracy: {best_accuracy}")
    '''
    
    trainer = Trainer(model_name=model_name)
    trainer.train()
    trainer.plot_losses()

if __name__ == "__main__":
    main()

