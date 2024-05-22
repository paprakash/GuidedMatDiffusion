import torch
import torch.nn as nn

class AdapterModule(nn.Module):
    def __init__(self, hidden_dim, property_dim):
        super(AdapterModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.property_dim = property_dim
        
        # Adapter: two-layer MLP
        self.adapter_fc1 = nn.Linear(property_dim, hidden_dim)
        self.adapter_relu = nn.ReLU()
        self.adapter_fc2 = None  # This will be initialized dynamically

        # Mixin: Will be initialized dynamically
        self.mixin = None

    def forward(self, H_L, g, property_label=None):
        input_dim = H_L.shape[-1]
        
        # Dynamically initialize layers if not already done
        if self.adapter_fc2 is None:
            self.adapter_fc2 = nn.Linear(self.hidden_dim, input_dim)
        
        if self.mixin is None:
            self.mixin = nn.Linear(input_dim, input_dim, bias=False)
            nn.init.zeros_(self.mixin.weight)  # Zero-initialize mixin weights

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
hidden_dim = 64  # Example hidden dimension
property_dim = 32  # Example property dimension

adapter_module = AdapterModule(hidden_dim, property_dim)

# Dummy data for demonstration
H_L = torch.randn((1, 128))  # Example H_L tensor
g = torch.randn((1, property_dim))  # Example property vector
property_label = torch.tensor([1])  # Example property label, not null

output = adapter_module(H_L, g, property_label)
print(output)

