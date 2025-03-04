# Federated Learning with Resource Monitoring

A federated learning (FL) framework built with Flower, PyTorch, and Docker, featuring real-time resource tracking (CPU, network, energy) using `psutil`. Trains a neural network on MNIST across two clients and a server, hitting **99.14% accuracy** in 30 rounds—all while logging detailed per-task resource stats inside containers.

## Features

- **Federated Learning**: Implements Flower’s FedAvg strategy to train a CNN on MNIST in a distributed setup.
- **Dataset**: Uses MNIST (handwritten digits, 60k train, 10k test), partitioned across clients.
- **Resource Monitoring**: Tracks CPU usage, network TX (MB), and estimated energy (J/s) per task with `psutil`.
- **Dockerized**: Runs in isolated containers (1 server, 2 clients) with GPU support (NVIDIA CUDA 12.4.1).
- **Real-Time**: Logs stats at key steps (Pre/Post-Training, Eval) every ~6-7s round.
- **Scalable**: Ready for DRL-based client selection and resource allocation (e.g., PPO).

## Results

- **Accuracy**: Centralized 99.14%, distributed ~98.87% (30 rounds, 1 epoch/round).
- **Runtime**: ~293s total (~9.8s/round).
- **Resources**:
  - Server: CPU ~11-12% (Pre), TX 0-21.31 MB.
  - Clients: CPU 0-38.5% (Pre), ~11-12% (Post), TX 0-5.46 MB.

Sample log:

2025-03-03 23:13:00 - Client 0 Post-Training - CPU: 11.7% | Freq: 2304 MHz | TX Power: 5.28 MB | Energy: 1.17 J/s
2025-03-03 23:13:00 - Server Round 30 Post-Eval - CPU: 6.4% | TX Power: 21.31 MB | Energy: 0.64 J/s


## Prerequisites

- Docker & Docker Compose
- NVIDIA GPU + CUDA drivers
- Python 3.9 (in containers)

## Setup

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/yourusername/fl-resource-monitor.git
   cd fl-resource-monitor

2. **Build Base Image (~55 mins first time)**:
   ```bash
   docker build -f base.Dockerfile -t fl_base:latest .

3. **Run**:
   ```bash
   docker-compose up --build

- Server: server:8080
- Clients: client1-1 (ID 0), client2-1 (ID 1)
4. **Logs**: Check docker-compose output for resource stats and accuracy.

## Project Structure

```
fl-resource-monitor/
├── base.Dockerfile         # Base image with dependencies
├── requirements.txt        # Python packages (flwr, torch, psutil)
├── server/
│   ├── Dockerfile         # Server container
│   ├── server.py          # FL server with FedAvg
│   ├── dataset.py         # MNIST data prep
│   └── model.py           # Neural net
├── client/
│   ├── Dockerfile         # Client container
│   ├── client.py          # FL client with resource logging
│   ├── dataset.py         # Same as server
│   └── model.py           # Same as server
├── docker-compose.yml      # Orchestrates 1 server, 2 clients
└── data/                   # MNIST data (auto-downloaded)
```
## How It Works
- Server: Initializes model, runs 30 FL rounds, aggregates client updates, and evaluates centrally.
- Clients: Train on MNIST partitions, report resources (`psutil`), and send updates.
- Resources: Logged via `psutil`—CPU %, TX power (MB), and energy (J/s) per task.

## Future Enhancements
- DRL: Add DRL for dynamic client selection and resource allocation (state: CPU, TX; actions: select, CPU %).
- Energy: Refine with real GPU wattage (e.g., 150W).
- Scale: Extend to 10+ clients with hierarchical FL.
