import os

import torch
from torch import optim

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, num_epochs=10, learning_rate=0.001,checkpoint_dir='checkpoint'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.lr=learning_rate
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.checkpoint_dir = checkpoint_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def save_checkpoint(self, epoch, best=False):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pt' if not best else 'best_model.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint '{path}' (epoch {checkpoint['epoch']})")

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        with torch.no_grad():  # Disable gradient calculation
            for inputs, labels in self.val_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def fine_tuning(self):
        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f"Epoch {epoch+1}/{self.num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            self.save_checkpoint(epoch)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, best=True)

    def evaluate_prompt(self, input_ids):
        self.model.eval()
        #input_ids = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)['input_ids']
        with torch.no_grad():
            output = self.model(input_ids)
        return output