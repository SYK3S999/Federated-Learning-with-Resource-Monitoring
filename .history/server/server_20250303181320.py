import flwr as fl
import torch
from collections import OrderedDict
from model import Net, test
from dataset import get_mnist, prepare_dataset

# Contribution tracker
class ContributionTracker:
    def __init__(self):
        self.contributions = {}
    def update(self, client_id, accuracy):
        self.contributions[client_id] = accuracy
    def get_top_clients(self, num_clients):
        sorted_clients = sorted(self.contributions.items(), key=lambda x: x[1], reverse=True)
        return [client_id for client_id, _ in sorted_clients[:num_clients]]

tracker = ContributionTracker()

# Config for clients
def get_on_fit_config():
    def fit_config_fn(server_round: int):
        return {"lr": 0.01, "momentum": 0.9, "local_epochs": 1}
    return fit_config_fn

# Global evaluation
def get_evaluate_fn(num_classes, testloader):
    def evaluate_fn(server_round, parameters, config):
        model = Net(num_classes)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        loss, accuracy = test(model, testloader, device)
        return loss, {"accuracy": accuracy}
    return evaluate_fn

# Custom strategy
class ContributionAndDiversityBasedFedAvg(fl.server.strategy.FedAvg):
    def configure_fit(self, server_round, parameters, client_manager):
        available_clients = list(client_manager.clients.keys())
        if server_round == 1:
            selected_clients = available_clients[:self.min_fit_clients]  # Placeholder diversity logic
        else:
            selected_clients = tracker.get_top_clients(self.min_fit_clients)
        print(f"Selected clients for round {server_round}: {selected_clients}")
        config = super().configure_fit(server_round, parameters, client_manager)
        return [(client, fit_config) for client, fit_config in config if client.cid in selected_clients]

    def aggregate_evaluate(self, server_round, results, failures):
        for cid, res in results:
            tracker.update(cid, res.metrics["accuracy"])
        return super().aggregate_evaluate(server_round, results, failures)

if __name__ == "__main__":
    # Load test data
    _, _, testloader = prepare_dataset(num_partitions=2, batch_size=128)  # Minimal partitions for server
    # Strategy
    strategy = ContributionAndDiversityBasedFedAvg(
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=get_on_fit_config(),
        evaluate_fn=get_evaluate_fn(num_classes=10, testloader=testloader),
        initial_parameters=fl.common.ndarrays_to_parameters([val.cpu().numpy() for _, val in Net(10).state_dict().items()])
    )
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy
    )