import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


class getDataset():

    def __init__(self):
        self.value = 4

    def getDataLoader(self, inputs, labels):
        labels = [
            [float(score) if score else 0 for score in sublist]
            for sublist in labels
        ]
        input_ids=inputs['input_ids']
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        dataset = TensorDataset(input_ids, labels_tensor)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)

        return train_loader, val_loader
