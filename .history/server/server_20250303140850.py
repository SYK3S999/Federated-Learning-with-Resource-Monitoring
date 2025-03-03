import flwr as fl
import torch
import numpy as np

# Define a simple PyTorch model
class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = torch.nn.Linear(784, 10)
        self.fc2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the global model
global_model = SimpleNet()

def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, weights):
    state_dict = {k: torch.from_numpy(np.array(v)) for k, v in zip(model.state_dict().keys(), weights)}
    model.load_state_dict(state_dict)

# Flower strategy (FedAvg)
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,  # Minimum clients to start training
    min_available_clients=2,  # Minimum clients to proceed
    on_fit_config_fn=lambda rnd: {"round": rnd},  # Pass config to clients
    initial_parameters=fl.common.ndarrays_to_parameters(get_weights(global_model)),
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),  # Run for 3 rounds
    strategy=strategy,
)