from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from helper import Model, WildBgDataSet, save_model


class ContactModel(Model):

    def __init__(self):
        super().__init__()
        self.num_inputs = 202

        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Linear(self.num_inputs, 300)
        self.hidden2 = nn.Linear(300, 250)
        self.hidden3 = nn.Linear(250, 200)

        # Output layer, 6 outputs for win/lose - normal/gammon/bg
        self.output = nn.Linear(200, 6)

        # Define activation function
        self.activation = nn.Hardsigmoid()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.hidden3(x)
        x = self.activation(x)
        x = self.output(x)
        return x


def train(model: Model):
    # Import rollout data
    rollout_data = WildBgDataSet("./training-data/contact-inputs.csv")
    train_loader = DataLoader(rollout_data, batch_size=64, shuffle=True)

    # "mps" takes more time than "cpu" on Macs, so let's ignore it for now.
    device = "cpu"
    print(f"Using {device} device")

    # CrossEntropyLoss is the best for a multi-class classifier. The model has only logits as outputs,
    # we add softmax later when we save the model to the disk.
    # CrossEntropyLoss supports soft probability labels in PyTorch 2.x
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1130e-6)

    model = model.to(device)

    for epoch in range(300):
        epoch_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            # set optimizer to zero grad to remove previous epoch gradients
            optimizer.zero_grad()
            # forward propagation
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backward propagation
            loss.backward()
            # optimize
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader) / 64

        epoch_plus_one = epoch + 1
        print(f'[Epoch: {epoch_plus_one}] loss: {epoch_loss:.5f}')

        if epoch_plus_one > 4:
            # Save epochs for each iteration after the first couple of epochs have passed
            save_model(model, "./training-data/contact" + f"{epoch_plus_one:03}" + ".onnx")


def main():
    # Make the training process deterministic
    torch.manual_seed(0)

    path = "./training-data/"
    Path(path).mkdir(exist_ok=True)

    model = ContactModel()
    train(model)

    print('Finished Training')


if __name__ == "__main__":
    main()
