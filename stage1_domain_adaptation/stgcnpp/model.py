import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Graph:
    """
    Constructs the Adjacency Matrix (A) for the 13-joint skeleton.
    Strategy: 'uniform' (1, V, V)
    """
    def __init__(self, num_node=13, max_hop=1):
        self.num_node = num_node
        self.max_hop = max_hop
        
        # 13-Joint Connections (Matched to dataset indices)
        self.edges = [
            (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), 
            (7, 8), (7, 9), (9, 11), (8, 10), (10, 12), (0, 1), (0, 2)
        ]
        
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        # 1. Build Base Adjacency
        adj = np.zeros((self.num_node, self.num_node))
        for i, j in self.edges:
            adj[i, j] = 1
            adj[j, i] = 1
            
        # 2. Normalize
        # (Uniform Strategy: A single normalized matrix including self-loops)
        # Add self-loops
        adj_with_eye = adj + np.eye(self.num_node)
        
        # Row-normalize (D^-1 A)
        Dl = np.sum(adj_with_eye, 0)
        Dn = np.zeros((self.num_node, self.num_node))
        for i in range(self.num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        
        AD = np.dot(adj_with_eye, Dn)
        
        # Reshape to (1, V, V) for the model
        return np.expand_dims(AD, 0)

class STGCNPlusPlusSpatial(nn.Module):
    def __init__(self, in_channels, out_channels, A, layer_norm=False):
        super().__init__()
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        self.num_subset = A.shape[0]
        
        self.conv = nn.Conv2d(in_channels, out_channels * self.num_subset, kernel_size=1)
        
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()
        x_conv = self.conv(x)
        x_conv = x_conv.view(N, self.num_subset, -1, T, V)
        # Graph Conv: Sum_k (X_k * A_k)
        y = torch.einsum('nkctv,kvw->nctw', x_conv, self.PA)
        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)

class MSTCN(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, 1)
        
        self.branch2 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1)
        )

        mid_c = out_channels // 4
        self.branch_convs = nn.ModuleList()
        self.transform = nn.Conv2d(in_channels, mid_c * 4, 1)
        
        dilations = [1, 2, 3, 4]
        for d in dilations:
            pad = ((3 + (3-1)*(d-1)) // 2, 0)
            self.branch_convs.append(nn.Sequential(
                nn.Conv2d(mid_c, mid_c, kernel_size=(3,1), stride=(stride,1), 
                          padding=pad, dilation=(d,1)),
                nn.BatchNorm2d(mid_c),
                nn.ReLU()
            ))

        self.agg = nn.Conv2d(out_channels * 2 + mid_c * 4, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        b1 = self.branch1(x)
        if self.branch2[0].stride[0] > 1:
             b1 = F.avg_pool2d(b1, kernel_size=(3,1), stride=(2,1), padding=(1,0))

        b2 = self.branch2(x)
        
        x_t = self.transform(x)
        x_splits = torch.chunk(x_t, 4, dim=1)
        b_results = [conv(split) for conv, split in zip(self.branch_convs, x_splits)]
        
        out = torch.cat([b1, b2] + b_results, dim=1)
        out = self.agg(out)
        out = self.bn(out)
        return self.relu(out + self.residual(x))

class STGCNPlusPlusBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1):
        super().__init__()
        self.gcn = STGCNPlusPlusSpatial(in_channels, out_channels, A)
        self.tcn = MSTCN(out_channels, out_channels, stride=stride)
        
    def forward(self, x):
        x = self.gcn(x)
        x = self.tcn(x)
        return x

class STGCNPlusPlus(nn.Module):
    def __init__(self, num_classes, in_channels=2, num_point=13, num_frame=100, base_channels=64):
        super().__init__()
        self.num_point = num_point
        self.num_frame = num_frame
        
        graph = Graph(num_node=num_point)
        self.A = graph.A
        
        self.data_bn = nn.BatchNorm1d(in_channels * num_point)
        self.layers = nn.ModuleList()
        
        # Layers config (ST-GCN++ style)
        cfgs = [
            # in, out, stride
            (in_channels, base_channels, 1),
            (base_channels, base_channels, 1),
            (base_channels, base_channels, 1),
            (base_channels, base_channels*2, 2), # Downsample time
            (base_channels*2, base_channels*2, 1),
            (base_channels*2, base_channels*2, 1),
            (base_channels*2, base_channels*4, 2), # Downsample time
            (base_channels*4, base_channels*4, 1),
            (base_channels*4, base_channels*4, 1)
        ]

        for i, (inc, outc, s) in enumerate(cfgs):
            self.layers.append(STGCNPlusPlusBlock(inc, outc, self.A, stride=s))

        self.fc = nn.Linear(base_channels*4, num_classes)

    def forward(self, data):
        # data.x shape: [Total_Nodes, C] = [Batch * T * V, C]
        x = data.x
        batch_size = data.num_graphs
        V = self.num_point
        T = self.num_frame
        C = x.shape[1]

        # Safety Check: Ensure the data aligns with the fixed dimensions
        # If this asserts, your dataset might have samples with different frame counts
        assert x.shape[0] == batch_size * T * V, \
            f"Input shape mismatch! Expected {batch_size*T*V} nodes, got {x.shape[0]}"

        # Unbatching: [B*T*V, C] -> [B, T, V, C] -> [B, C, T, V]
        # This view() is safe because PyG DataLoader creates batches by simple concatenation.
        # It does NOT shuffle frames internally within a batch.
        x = x.view(batch_size, T, V, C).permute(0, 3, 1, 2).contiguous()
        
        # Data BN
        N, C, T, V = x.size()
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()
        
        for layer in self.layers:
            x = layer(x)
            
        # Global Pooling
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(N, -1)
        
        return self.fc(x)