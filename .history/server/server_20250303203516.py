import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
from model import Net, test
from dataset import prepare_dataset
import torch

def evaluate_metrics_aggregation_fn(eval_metrics):
    accuracies = [m["accuracy"] for _, m in eval_metrics]
    return {"accuracy": sum(accuracies) / len(accuracies)}

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "lr": 0.01,           # Learning rate
        "momentum": 0.9,      # Momentum for SGD
        "local_epochs": 1,    # Number of local epochs
    }
    return config

if __name__ == "__main__":
    _, _, testloader = prepare_dataset(num_partitions=2, batch_size=128)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net(10).to(device)
    
    def evaluate_fn(server_round, parameters, config):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        net.load_state_dict(state_dict, strict=True)
        loss, accuracy = test(net, testloader, device)
        return loss, {"accuracy": accuracy}

    # Convert initial parameters to Flower's format
    initial_params = [val.cpu().numpy() for _, val in net.state_dict().items()]
    initial_parameters = ndarrays_to_parameters(initial_params)

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=evaluate_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,  # Add config function
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )