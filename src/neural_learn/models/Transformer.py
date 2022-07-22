import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


"""
    Q, K, and V are batches of matrices, each with shape (batch_size, seq_length, num_features). 
    Multiplying the query (Q) and key (K) arrays results in a (batch_size, seq_length, seq_length) 
    array, which tells us roughly how important each element in the sequence is. 

    This is the attention of this layer — it determines which elements we “pay attention” to. 
    The attention array is normalized using softmax, so that all of the weights sum to one. 
    (Because we can’t pay more than 100% attention, right?) Finally, the attention is applied 
    to the value (V) array using matrix multiplication.
"""

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """
        Q, K, and V are batches of matrices, each with shape (batch_size, seq_length, num_features)
    """
    Q_KT = query.bmm(key.transpose(1,2))    # swaps dimensions: seq_length and Num_features

    # Square root of dk
    dk = query.size(2) ** 0.5
    
    # Softmax of this
    softmax = F.softmax(Q_KT / dk , dim=-1)

    # Multiply with value now
    output = softmax.bmm(value)

    return output

"""
    Each attention head contains 3 linear layers, followed by scaled dot-product attention. 
    Lets encapsulate this in an AttentionHead layer:
"""
class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


"""
    Now, its very easy to build the multi-head attention layer. 
    Just combine num_heads different attention heads and a Linear layer for the output.
"""

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()

        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )

if __name__ == "__main__":
    import os
    try:
        os.system("clear")
    except:
        pass

    # Query Key Value
    q = torch.rand(8, 50, 6)
    k = torch.rand(8, 50, 6)
    v = torch.rand(8, 50, 6)

    output = scaled_dot_product_attention(q, k, v)
    print(output.shape)