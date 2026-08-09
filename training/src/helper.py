from abc import abstractmethod, ABC
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch


class WildBgDataSet(Dataset):
    def __init__(self, csv_files: list | str):
        if isinstance(csv_files, str):
            csv_files = [csv_files]
        labels = []
        inputs = []
        for path in csv_files:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    line = line.strip().split(',')
                    line = list(map(float, line))
                    labels.append(line[:6])
                    inputs.append(line[6:])
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class Model(torch.nn.Module, ABC):
    # Override that in implementing classes
    num_inputs = 0


# Wrap a model with logits as output and add softmax so that all outputs add up to 1.
class Wrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.base_model(x)
        return self.softmax(logits)


def save_model(model: Model, path: str) -> None:
    wrapper = Wrapper(model)
    dummy_input = torch.randn(1, model.num_inputs, requires_grad=True)
    torch.onnx.export(wrapper, dummy_input, path)
