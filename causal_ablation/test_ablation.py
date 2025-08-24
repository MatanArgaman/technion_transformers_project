import torch
import torch.nn as nn

# Simple hook system with handle
class HookHandle:
    def __init__(self, hook_point, hook_fn):
        self.hook_point = hook_point
        self.hook_fn = hook_fn
        self.removed = False

    def remove(self):
        if not self.removed:
            self.hook_point.hooks.remove(self.hook_fn)
            self.removed = True

class HookPoint:
    def __init__(self):
        self.hooks = []

    def add_hook(self, hook_fn):
        self.hooks.append(hook_fn)
        return HookHandle(self, hook_fn)

    def __call__(self, tensor):
        for hook in self.hooks:
            tensor = hook(tensor, self)
        return tensor

# Simple MLP model with hook
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=3, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.hook_mlp_out = HookPoint()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Add hook_dict for compatibility
        self.hook_dict = {'hook_mlp_out': self.hook_mlp_out}

    def forward(self, x):
        x = self.fc1(x)
        x = self.hook_mlp_out(x)
        x = self.fc2(x)
        return x

# Utility function to print model weights by layer
def print_model_weights(model):
    print("Model weights by layer:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")

# Test function
def test_neuron_ablation():
    torch.manual_seed(0)
    model = SimpleMLP()
    print_model_weights(model)
    x = torch.randn(1, 4)
    print("Input:", x)

    def print_hidden_after_ablation(model, x, num):
        with torch.no_grad():
            hidden = model.hook_mlp_out(model.fc1(x))
            print(f"Hidden after ablation of neuron {num}:", hidden)
    
    def print_hidden_after_returning(model, x, num):
        with torch.no_grad():
            hidden = model.hook_mlp_out(model.fc1(x))
            print(f"Hidden after return of neuron {num}:", hidden)


    # Forward pass without ablation
    out1 = model(x)
    print("Output before ablation:", out1)
    
    # Print hidden activations before ablation
    with torch.no_grad():
        hidden = model.fc1(x)
        print("Hidden before ablation:", hidden)

    # Define ablation hook for neuron 1
    def ablation_hook1(tensor, hook):
        tensor = tensor.clone()
        tensor[..., 0] = 0
        return tensor

    # Register ablation hook for neuron 1 and get handle
    handle1 = model.hook_dict['hook_mlp_out'].add_hook(ablation_hook1)

    print_hidden_after_ablation(model, x, 1)

    # print_model_weights(model)

    # Forward pass with neuron 1 ablated
    out2 = model(x)
    print("Output after ablation of neuron 1:", out2)
    assert not torch.allclose(out1, out2), "Ablation of neuron 1 did not change the output!"

    # Define ablation hook for neuron 2
    def ablation_hook2(tensor, hook):
        tensor = tensor.clone()
        tensor[..., 2] = 0  
        return tensor

    # Register ablation hook for neuron 2 and get handle
    handle2 = model.hook_dict['hook_mlp_out'].add_hook(ablation_hook2)

    print_hidden_after_ablation(model, x, 2)

    # Forward pass with neuron 1 and 2 ablated
    out3 = model(x)
    print("Output after ablation of neuron 1 and 2:", out3)
    assert not torch.allclose(out2, out3), "Ablation of neuron 1 and 2 did not change the output!"

    # Remove both hooks
    handle2.remove()
    out4= model(x)
    print("Output after removing ablation hook for neuron 2:", out4)
    print_hidden_after_returning(model, x, 2)

    handle1.remove()
    print_hidden_after_returning(model, x, 1)

    # Forward pass after removing both hooks
    out5= model(x)
    print("Output after removing both ablation hooks:", out5)
    assert torch.allclose(out1, out5), "Output did not return to original after removing both hooks!"

    print("Test passed: Output changed after each ablation and returned to original after removing both hooks.")

if __name__ == "__main__":
    test_neuron_ablation() 