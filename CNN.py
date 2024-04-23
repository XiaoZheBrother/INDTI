import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN(nn.Sequential):
    def __init__(self, encoding, cnn_filters, cnn_kernels, hidden_dim):
        super(CNN, self).__init__()
        if encoding == 'drug':
            in_ch = [63] + cnn_filters
            kernels = cnn_kernels
            layer_size = len(cnn_filters)
            self.conv = nn.ModuleList([nn.Conv1d(in_channels=in_ch[i],
                                                 out_channels=in_ch[i + 1],
                                                 kernel_size=kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double()
            n_size_d = self._get_conv_output((63, 100))
            # n_size_d = 1000
            self.fc1 = nn.Linear(n_size_d, hidden_dim)

        elif encoding == 'protein':
            in_ch = [21] + cnn_filters
            kernels = cnn_kernels
            layer_size = len(cnn_filters)
            self.conv = nn.ModuleList([nn.Conv1d(in_channels=in_ch[i],
                                                 out_channels=in_ch[i + 1],
                                                 kernel_size=kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double()
            n_size_p = self._get_conv_output((21, 1000))

            self.fc1 = nn.Linear(n_size_p, hidden_dim)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, v):
        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v


class MLP(nn.Sequential):
    def __init__(self, input_dim, output_dim, hidden_dims_lst):
        """
            input_dim (int)
            output_dim (int)
            hidden_dims_lst (list, each element is a integer, indicating the hidden size)
        """
        super(MLP, self).__init__()
        layer_size = len(hidden_dims_lst) + 1
        dims = [input_dim] + hidden_dims_lst + [output_dim]
        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v):
        # predict
        v = v.float()
        for i, l in enumerate(self.predictor):
            v = F.relu(l(v))
        return v
