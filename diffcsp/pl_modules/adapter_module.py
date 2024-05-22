import torch
import torch.nn as nn

class AdapterModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, property_dim):  # input_dim=dim of hidden variable, and property_dim=property embedding
        super(AdapterModule, self).__init__()
        # Adapter is a two-layer MLP
        self.adapter_fc1 = nn.Linear(property_dim, hidden_dim)
        self.adapter_relu = nn.ReLU() # not sure if matergen does this but it seems a good idea
        self.adapter_fc2 = nn.Linear(hidden_dim, input_dim)
        
        # Mixin is a zero-initialized linear layer without bias
        self.mixin = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.zeros_(self.mixin.weight)  # Zero-initialize mixin weights

    def forward(self, H_L, g, property_label=None):
        if property_label is not None:
            f_adapter_L = self.adapter_fc1(g)
            f_adapter_L = self.adapter_relu(f_adapter_L)
            f_adapter_L = self.adapter_fc2(f_adapter_L)
            
            f_mixin_L = self.mixin(f_adapter_L)
            
            H_prime_L = H_L + f_mixin_L
        else:
            H_prime_L = H_L

        return H_prime_L

# Example usage
input_dim = 128  # Example dimension
hidden_dim = 64  # Example hidden dimension
property_dim = 32  # Example property dimension

adapter_module = AdapterModule(input_dim, hidden_dim, property_dim)

# Dummy data for demonstration
H_L = torch.randn((1, input_dim))
g = torch.randn((1, property_dim))
property_label = torch.tensor([1])  # Example property label, not null

output = adapter_module(H_L, g, property_label)
print(output)

