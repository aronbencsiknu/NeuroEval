import torch
import torch.optim as optim
import snntorch.functional as SF  # Ensure this module is correctly imported
from . import utils

indices = None

class Trainer:
    def __init__(self, net, dataloader, graph, target_sparcity, num_epochs=150, learning_rate=1e-4, target_frequency=0.5, batch_size=16, num_steps=10):
        self.net = net
        self.dataloader = dataloader
        self.graph = graph
        self.target_sparcity = target_sparcity
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.target_frequency = target_frequency
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        self.criterion = SF.ce_count_loss()

        self.xor_inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        #self.xor_targets = torch.tensor([[0], [1], [1], [0]], dtype=torch.int32)
        self.xor_targets = torch.tensor([[0], [1], [1], [0]])
        self.spike_record = {}

        print("\n----- TRAINING -----\n")
    
    def generate_spike_train(self, input_data, num_steps, spike_prob=0.5):
        spike_train = torch.zeros((num_steps, input_data.size(0), input_data.size(1)))
        for t in range(num_steps):
            spike_train[t] = (torch.rand(input_data.size()) < (input_data * spike_prob)).float()

        return spike_train

    def train(self, device, mapping, dut=None):
        self.net = self.net.to(device)

        global indices

        indices = mapping.indices_to_lock
        num_long_range_conns, num_short_range_conns = utils.calculate_lr_sr_conns(mapping, self.graph)
        ratio = num_long_range_conns / (num_long_range_conns + num_short_range_conns)
        
        print("lr:",num_long_range_conns,"// sr:",num_short_range_conns)

        conn_reps = int((num_long_range_conns - num_long_range_conns * self.target_sparcity)/self.num_epochs)

        print("CONN REPS", conn_reps)

        def zero_out_grads(grad):
            grad = grad.clone()
            for idx in indices["indices"]:
                grad[idx] = 0
            return grad
        
        def test(module, input, output):
             global indices
             for idx in indices["indices"]:
                module.weight.data[idx] = 0

        self.net.lif1.recurrent.register_forward_hook(test)

        if indices is not None:
            self.net.lif1.recurrent.weight.register_hook(zero_out_grads)

        for epoch in range(self.num_epochs):

            for data, target in self.dataloader:
                data = data.to(device)
                target = target.to(device)
                # num_long_range_conns, num_short_range_conns = utils.calculate_lr_sr_conns(mapping, self.graph)
                # ratio = num_long_range_conns / (num_long_range_conns + num_short_range_conns)
                
                # print("lr:",num_long_range_conns,"// sr:",num_short_range_conns)
                # print("RATIO LR", ratio)

                mapping = utils.choose_conn_remove(mapping, reps=conn_reps)
                # indices = mapping.indices_to_lock
                
                # # mock input
                # input_indices = torch.randperm(4)
                # inputs = self.xor_inputs[input_indices]
                # target = self.xor_targets[input_indices].to(device)

                # Generate spike trains
                #inputs = self.generate_spike_train(inputs, self.num_steps).to(device)

                # Forward pass
                outputs, _ = self.net(data, time_first=False)

                #print("Updated weights:\n", self.net.lif1.recurrent.weight.data[0])

                # Remove redundant dimension
                #target = target.squeeze(1)

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
        num_long_range_conns, num_short_range_conns = utils.calculate_lr_sr_conns(mapping, self.graph)
        
        print("lr:",num_long_range_conns,"// sr:",num_short_range_conns)
        
        return self.net, mapping
