import flwr as fl
import torch
from collections import OrderedDict
import os
import time
from client.dataset import prepare_dataset
from model import Net, test, train

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, valloader, num_classes):
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net(num_classes)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]
        scaler = torch.amp.GradScaler('cuda')  # Updated for PyTorch 2.4
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        train(self.model, self.trainloader, optim, epochs, self.device)
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": accuracy}

if __name__ == "__main__":
    cid = int(os.getenv("CLIENT_ID", 0))
    time.sleep(10)  # Wait for server
    trainloaders, valloaders, _ = prepare_dataset(num_partitions=2, batch_size=20)
    client = FlowerClient(trainloaders[cid], valloaders[cid], num_classes=10)
    fl.client.start_numpy_client(server_address="server:8080", client=client)