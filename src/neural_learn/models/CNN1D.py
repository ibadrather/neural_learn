
import torch.nn as nn

class CNN1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        n_targets: number of classes
        
    """
    def __init__(self, 
                n_features,
                n_targets,
                window_size,
                feature_dim=64,
                kernel_size=6,
                stride=1,
                activation=nn.Relu(),
                dropout=0.2,
                softmax_output=False, 
                verbose=False):
        super(CNN1D, self).__init__()

        self.n_features = n_features
        self.n_targets = n_targets
        self.activation = activation

        # if we want a softmax at the output
        self.softmax_output = softmax_output

        self.conv1 = nn.Sequential(
                            nn.Conv1d(self.n_features, feature_dim, kernel_size=kernel_size, 
                                      padding=kernel_size//2, stride=stride, bias=True), 
                            nn.BatchNorm1d(feature_dim, feature_dim),
                            self.activation
                            )
        
        self.conv2 = nn.Sequential(
                            nn.Conv1d(feature_dim, 2*feature_dim, kernel_size=3, 
                                      padding=1, stride=stride, bias=True), 
                            nn.BatchNorm1d(2*feature_dim, 2*feature_dim),
                            self.activation
                            )

        self.conv3 = nn.Sequential(
                            nn.Conv1d(2*feature_dim, 2*feature_dim, kernel_size=3, 
                                      padding=1, stride=stride, bias=True), 
                            nn.BatchNorm1d(2*feature_dim, 2*feature_dim),
                            self.activation
                            )

        self.conv4 = nn.Sequential(
                            nn.Conv1d(2*feature_dim, 2*feature_dim, kernel_size=3, 
                                      padding=1, stride=stride, bias=True), 
                            nn.BatchNorm1d(2*feature_dim, 2*feature_dim),
                            self.activation
                            )
        
        self.conv5 = nn.Sequential(
                            nn.Conv1d(2*feature_dim, 4*feature_dim, kernel_size=3, 
                                      padding=1, stride=stride, bias=True), 
                            nn.BatchNorm1d(4*feature_dim, 4*feature_dim),
                            self.activation
                            )

        self.conv6 = nn.Sequential(
                            nn.Conv1d(4*feature_dim, 4*feature_dim, kernel_size=3, 
                                      padding=1, stride=stride, bias=True), 
                            nn.BatchNorm1d(4*feature_dim, 4*feature_dim),
                            self.activation
                            )

        self.dropout = nn.Dropout(dropout, inplace=False)
        
        self.fc1 = nn.Sequential(
                        nn.Linear(4*feature_dim, 8*feature_dim, bias=True),
                        self.activation,
                        self.dropout
                                )
        self.fc2 = nn.Sequential(
                        nn.Linear(8*feature_dim, 8*feature_dim, bias=True),
                        self.activation,
                        self.dropout
                                )

        self.output = nn.Linear(8*feature_dim, self.n_targets, bias=True)                                              
     
        self._initialize()


    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight,0.1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)

        out = self.fc1(out)
        out = self.fc2(out)

        output = self.output(out)

        if self.softmax_output:
            return nn.Softmax(output, dim=1)
        
        return output

    

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