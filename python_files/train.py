import torch
import torch.optim as optim
import snntorch.functional as SF  # Ensure this module is correctly imported
from . import utils

class Trainer:
    def __init__(self, net, mapping, graph, num_epochs=150, learning_rate=1e-4, target_frequency=0.5, batch_size=16, num_steps=10):
        self.net = net
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.target_frequency = target_frequency
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        self.criterion = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.5)

        self.xor_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        self.xor_targets = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
        self.spike_record = {}

        self.mapping = mapping
        self.graph = graph

        print("\n----- TRAINING -----\n")
    
    def generate_spike_train(self, input_data, num_steps, spike_prob=0.5):
        spike_train = torch.zeros((num_steps, input_data.size(0), input_data.size(1)))
        for t in range(num_steps):
            spike_train[t] = (torch.rand(input_data.size()) < (input_data * spike_prob)).float()

        return spike_train

    def train(self, device, dut=None):
        self.net = self.net.to(device)
        self.xor_inputs = self.xor_inputs
        self.xor_targets = self.xor_targets

        num_long_range_conns, num_short_range_conns = utils.calculate_lr_sr_conns(self.mapping, self.graph)

        ratio = num_long_range_conns / (num_long_range_conns + num_short_range_conns)
        print("RATIO LR", ratio)

        for epoch in range(self.num_epochs):
            indices = torch.randperm(4)
            inputs = self.xor_inputs[indices]
            target = self.xor_targets[indices].to(device)

            # Generate spike trains
            inputs = self.generate_spike_train(inputs, self.num_steps).to(device)

            # Forward pass
            outputs, _ = self.net(inputs)

            # Define target firing frequency
            target = target.squeeze(1)

            # Calculate loss
            loss = self.criterion(outputs, target)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print loss
            if (epoch + 1) % 10 == 0:
                temp = f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}"
                
                if dut is not None:
                    dut._log.info(temp)
                else:
                    print(temp)

        return self.net
