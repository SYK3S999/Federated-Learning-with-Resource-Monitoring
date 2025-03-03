import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
from model import Net, test
from dataset import prepare_dataset
import torch

def evaluate_metrics_aggregation_fn(eval_metrics):
    accuracies = [m["accuracy"] for _, m in eval_metrics]
    return {"accuracy": sum(accuracies) / len(accuracies)}

def fit_metrics_aggregation_fn(fit_metrics):
    accuracies = [m.get("accuracy", 0) for _, m in fit_metrics]
    return {"accuracy": sum(accuracies) / len(accuracies) if accuracies else 0}

def fit_config(server_round: int):
    return {"lr": 0.01, "momentum": 0.9, "local_epochs": 1}

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
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,  # Added
        on_fit_config_fn=fit_config,
        initial_parameters=initial_parameters,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
    )