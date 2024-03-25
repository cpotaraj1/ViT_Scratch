import torch
from torch import nn, optim
import os

from utils.model_architecture import ViTForClassfication
from utils.experiment import save_checkpoint, save_experiment, add_attention_tboard
from utils.data_prep import prepare_data

from torch.utils.tensorboard import SummaryWriter


exp_name = 'vit_100_'
batch_size = 32
epochs = 1000
lr = 1e-2
save_model_every = 20
writer = SummaryWriter(f'runs/{exp_name}')
tboard = f'runs/{exp_name}/tb_images'
if not os.path.exists(tboard):
    os.makedirs(tboard)


device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 384,
    "num_hidden_layers": 6,
    "num_attention_heads": 6,
    "intermediate_size": 4 * 384, # 4 * hidden_size
    "hidden_dropout_prob": 0.2,
    "attention_probs_dropout_prob": 0.2,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}

# These are not hard constraints, but are used to prevent misconfigurations
assert config["hidden_size"] % config["num_attention_heads"] == 0
assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0


class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device
        self.best_val_loss = float('inf')

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        # Train the model
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            writer.add_scalar('Training loss', train_loss, i)
            writer.add_scalar('Validation loss', test_loss, i)
            writer.add_scalar('Accuracy', accuracy, i)
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch', i+1)
                save_checkpoint(self.exp_name, self.model, i+1)
            
            if test_loss < self.best_val_loss:
                print(f"Best model found at epoch: {i}")
                save_checkpoint(self.exp_name, self.model, "best")
                self.best_val_loss = test_loss

            
        # Save the experiment
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        for batch in trainloader:
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(images)[0], labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                batch = [t.to(self.device) for t in batch]
                images, labels = batch
                logits, _ = self.model(images)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


def main():
    save_model_every_n_epochs = save_model_every
    trainloader, testloader, _ = prepare_data(batch_size=batch_size)
    model = ViTForClassfication(config)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, exp_name, device=device)
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)


if __name__ == '__main__':
    main()