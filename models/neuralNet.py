import torch


class NeuralNet(torch.nn.Module):
    """
    This class represents a simple Neural network with ReLU activations and BatchNorm
    """

    def __init__(self, input_size, output_size):
        super(NeuralNet, self).__init__()
        # self.linear = torch.nn.Linear(input_size, output_size)
        self.layers = torch.nn.Sequential(*
                                          [torch.nn.Linear(input_size, input_size), torch.nn.BatchNorm1d(input_size),
                                           torch.nn.ReLU(), torch.nn.Dropout(p=0.5),
                                           torch.nn.Linear(input_size, input_size), torch.nn.BatchNorm1d(input_size),
                                           torch.nn.ReLU(), torch.nn.Dropout(p=0.2),
                                           torch.nn.Linear(input_size, 10000), torch.nn.BatchNorm1d(10000),
                                           torch.nn.Linear(10000, 1)])

    def forward(self, x):
        return self.layers(x)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
