import torch


class RegressionNet(torch.nn.Module):
    """
    This class represents the Neural network which will be used to perform simple OLS regression
    """

    def __init__(self, input_size, output_size):
        super(RegressionNet, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
