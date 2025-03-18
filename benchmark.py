import torch
import torch.nn as nn
import time
from torch import optim
from utils.ReDo import ReDo, GradientReDo
import random
import numpy as np

# Set random seed to ensure experiment reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

# Benchmark function
def benchmark_redo(model, optimizer, num_steps=1000):
    redo = ReDo(model, frequency=10, optimizer=optimizer)
    for _ in range(num_steps):
        x = torch.randn(100, 10)
        loss = torch.nn.functional.mse_loss(model(x), torch.randn(100, 10))
        optimizer.zero_grad()
        loss.backward()
        redo.step(x)
        optimizer.step()

def benchmark_grad_redo(model, optimizer, num_steps=1000):
    grad_redo = GradientReDo(model, frequency=10, optimizer=optimizer)
    for _ in range(num_steps):
        x = torch.randn(100, 10)
        loss = torch.nn.functional.mse_loss(model(x), torch.randn(100, 10))
        optimizer.zero_grad()
        loss.backward()
        grad_redo.step()
        optimizer.step()

# Main program
if __name__ == "__main__":
    set_seed(42)  # Set random seed to ensure reproducibility

    model = SimpleModel()
    optimizer = optim.Adam(model.parameters())

    # Test ReDo
    start_time = time.time()
    benchmark_redo(model, optimizer)
    redo_time = time.time() - start_time

    # Test GradientReDo
    start_time = time.time()
    benchmark_grad_redo(model, optimizer)
    grad_redo_time = time.time() - start_time

    print(f"ReDo time: {redo_time:.4f} seconds\n")
    print(f"GradientReDo time: {grad_redo_time:.4f} seconds\n")