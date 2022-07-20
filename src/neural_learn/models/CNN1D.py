import torch.nn as nn


class IMUCLSBaseline(nn.Module):
    def __init__(self, n_features,
                n_targets,
                window_size,
                feature_dim=64,
                kernel_size=6,
                dropout=0.2,
                softmax_output=False):

        super(IMUCLSBaseline, self).__init__()

        self.softmax_output = softmax_output

        self.conv1 = nn.Sequential(
                            nn.Conv1d(n_features, feature_dim, kernel_size=kernel_size), 
                            nn.ReLU()
                            )
        self.conv2 = nn.Sequential(
                            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size), 
                            nn.ReLU()
                            )

        self.dropout = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool1d(2) # Collapse T time steps to T/2
        self.fc1 = nn.Linear(window_size*(feature_dim//2), feature_dim, nn.ReLU())
        self.fc2 = nn.Linear(feature_dim,  n_targets)
        self.softmax = nn.Softmax(dim=1)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        """
        Forward pass
        :param x:  B X M x T tensor reprensting a batch of size B of  M sensors (measurements) X T time steps (e.g. 128 x 6 x 100)
        :return: B X N weight for each mode per sample
        """
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.maxpool(x) # return B X C/2 x M
        x = x.view(x.size(0), -1) # B X C/2*M
        x = self.fc1(x)
        x = self.fc2(x)

        if self.softmax_output:
            return self.softmax(x)
        else:
            return x


if __name__ =="__main__":
    pass