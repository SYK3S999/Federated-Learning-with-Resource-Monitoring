import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import psutil
import time
import random

# Define the same PyTorch model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Simulated dataset
dummy_data = torch.randn(100, 784)
dummy_labels = torch.randint(0, 10, (100,))

# Resource measurement
def measure_resources():
    start_time = time.time()
    cpu_start = psutil.cpu_percent(interval=None)
    bytes_sent = random.randint(1000, 10000)  # Simulate bandwidth
    energy = (time.time() - start_time) * 0.1  # Simulate energy (simplified)
    return {"cpu_cycles": cpu_start, "bandwidth": bytes_sent, "energy": energy}

# Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = cid
        self.model = SimpleNet()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = {k: torch.from_numpy(np.array(v)) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        resources = measure_resources()
        print(f"Client {self.cid} training... Resources: {resources}")

        # Train locally
        self.model.train()
        for _ in range(1):  # 1 epoch
            self.optimizer.zero_grad()
            output = self.model(dummy_data)
            loss = self.criterion(output, dummy_labels)
            loss.backward()
            self.optimizer.step()

        return self.get_parameters(config), len(dummy_data), {"resources": resources}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            output = self.model(dummy_data)
            loss = self.criterion(output, dummy_labels)
        return float(loss), len(dummy_data), {}

# Start Flower client
def main():
    client_id = str(random.randint(1, 1000))
    client = FlowerClient(client_id)
    fl.client.start_numpy_client(server_address="server:8080", client=client)

if __name__ == "__main__":
    main()