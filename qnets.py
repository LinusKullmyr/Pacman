import torch
import torch.nn as nn
import time
from functools import wraps


class double_DQN:
    def __init__(self, input_channels, output_size, learning_rate, gamma):
        self.device = self._get_device()
        print("Using device: ", self.device)

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.policy_network = CNN(input_channels, output_size).to(self.device)
        self.target_network = CNN(input_channels, output_size).to(self.device)
        self.target_network.eval()  # Only evals from target network
        self.update_target_network()  # Do an inital sync

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        # Use wrapper to track time used
        self.times = {}
        self.predict_eval = self.timing(self.predict_eval)
        self.training_step = self.timing(self.training_step)

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def predict_eval(self, state_tensor):
        state_tensor = state_tensor.to(self.device)
        self.policy_network.eval()
        with torch.no_grad():
            action = self.policy_network(state_tensor).detach().cpu()

        return action

    def training_step(self, batch):
        states, actions, next_states, rewards, dones = (t.to(self.device) for t in batch)

        # Put policy network in training mode
        self.policy_network.train()

        # Predict current state Q-values from policy network
        current_q_values = self.policy_network(states).gather(1, actions).squeeze(-1)

        # Predict next state Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].detach()

        # mask out done states
        next_q_values[dones.squeeze(-1)] = 0.0

        # Calculate expected Q values
        expected_q_values = rewards.squeeze(-1) + self.gamma * next_q_values

        assert current_q_values.shape == expected_q_values.shape

        # Compute loss
        loss = nn.functional.mse_loss(current_q_values, expected_q_values)

        # backward prop and optimization
        self.optimizer.zero_grad(set_to_none=True)  # In-place zeroing of gradients
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu()

    def timing(self, func):
        """
        Decorator to measure the average execution time of methods.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()

            func_name = func.__name__
            if func_name not in self.times:
                self.times[func_name] = {"total_time": 0, "call_count": 0}

            elapsed_time = end_time - start_time
            self.times[func_name]["total_time"] += elapsed_time
            self.times[func_name]["call_count"] += 1

            return result

        return wrapper

    def get_times(self, func_name):
        """Returns the average execution time for the specified function."""
        if self.times[func_name]["call_count"] == 0:
            return 0
        calls = self.times[func_name]["call_count"]
        tot = self.times[func_name]["total_time"]
        avg = self.times[func_name]["total_time"] / self.times[func_name]["call_count"]
        return calls, tot, avg

    def reset_times(self):
        self.times = {}

    def _get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        return device


class CNN(nn.Module):
    def __init__(self, input_channels, output_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            #
            nn.Flatten(),
            #
            nn.LazyLinear(128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            #
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            #
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        return self.model(x)
