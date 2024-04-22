import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl


class DQN(pl.LightningModule):
    def __init__(self, input_channels, output_size, gamma, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.gamma = gamma

        # Initialize the policy and target networks
        self.policy_network = self.create_network(input_channels, output_size)
        self.target_network = self.create_network(input_channels, output_size)
        self.update_target_network()
        self.target_network.eval()  # Set target network to evaluation mode

    def create_network(self, input_channels, output_size):
        return nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.policy_network(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx=0):
        states, actions, rewards, next_states, dones = batch

        # Predict current state Q-values
        current_q_values = self.policy_network(states).gather(1, actions)

        # Predict next state Q-values
        next_q_values = self.target_network(next_states).max(1)[0].detach()  # No gradient tracking

        # Calculate expected Q values
        expected_q_values = rewards.squeeze(-1) + self.gamma * next_q_values * ~(dones.squeeze(-1))

        # Compute loss
        loss = nn.functional.mse_loss(current_q_values.squeeze(-1), expected_q_values)

        # Logging and optimization
        self.log("train_loss", loss)
        return loss

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
