import torch
import torch.optim as optim
import snntorch.functional as SF  # Ensure this module is correctly imported
from . import utils
#import wandb

indices = None

class Trainer:
    def __init__(self, net, train_loader, val_loader, target_sparcity, recall_duration, graph=None, num_epochs=150, learning_rate=1e-4, target_frequency=0.5, num_steps=10, optimizer="AdamW",wandb_logging=False):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.graph = graph
        self.target_sparcity = target_sparcity
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.target_frequency = target_frequency
        self.num_steps = num_steps
        self.recall_duration = recall_duration
        self.wandb_logging = wandb_logging

        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        elif optimizer == "AdamW":
            self.optimizer = optim.AdamW(self.net.parameters(), lr=learning_rate)
        # add exception handling

        self.criterion = SF.ce_count_loss()
        self.spike_record = {}

        print("\n----- TRAINING -----\n")
    
    def generate_spike_train(self, input_data, num_steps, spike_prob=0.5):
        spike_train = torch.zeros((num_steps, input_data.size(0), input_data.size(1)))
        for t in range(num_steps):
            spike_train[t] = (torch.rand(input_data.size()) < (input_data * spike_prob)).float()

        return spike_train
    

    def eval(self, device, val_idx):
        counter =  0
        acc = 0
        
        self.net.eval()
        with torch.no_grad():
            #print("Updated weights:\n", self.net.lif1.recurrent.weight.data)
            for data, target in self.val_loader:
                data = data.to(device)
                target = target.to(device)
                outputs, _ = self.net(data, time_first=False)
                acc += SF.acc.accuracy_rate(outputs, target)
                val_loss = self.criterion(outputs[-self.recall_duration:], target)
                if self.wandb_logging:
                    wandb.log({"Val loss": val_loss.item(),
                            "Val index": val_idx})
                counter += 1
                val_idx += 1
        
        return acc/counter, val_idx
    

    def train(self, device, mapping=None, dut=None):
        self.net = self.net.to(device)
        if self.target_sparcity != 1.0:
            global indices

            indices = mapping.indices_to_lock
            num_long_range_conns, num_short_range_conns = utils.calculate_lr_sr_conns(mapping, self.graph)
            #ratio = num_long_range_conns / (num_long_range_conns + num_short_range_conns)
            
            print("lr:",num_long_range_conns,"// sr:",num_short_range_conns)

            # how many connections to remove each (epoch?)
            conn_reps = int((num_long_range_conns - num_long_range_conns * self.target_sparcity)/(self.num_epochs))

            print("CONN REPS", conn_reps)

        def zero_out_grads(grad):
            grad = grad.clone()
            for idx in indices["indices"]:
                grad[idx] = 0
            return grad
        
        def zero_out_weights(module, input, output):
            global indices
            for idx in indices["indices"]:
                module.weight.data[idx] = 0

        # if indices is not None and self.target_sparcity != 1.0:
        #     self.net.lif1.recurrent.weight.register_hook(zero_out_grads)
        #     self.net.lif1.recurrent.register_forward_hook(zero_out_weights)

        train_index = 0
        val_index = 0
        accuracies = []
        for epoch in range(self.num_epochs):
            self.net.train()
            for data, target in self.train_loader:
                data = data.to(device)
                target = target.to(device)
                # num_long_range_conns, num_short_range_conns = utils.calculate_lr_sr_conns(mapping, self.graph)
                # ratio = num_long_range_conns / (num_long_range_conns + num_short_range_conns)
                
                # print("lr:",num_long_range_conns,"// sr:",num_short_range_conns)
                # print("RATIO LR", ratio)
                
                
                # Forward pass
                outputs, _ = self.net(data, time_first=False)

                # print("Updated weights:\n", self.net.lif1.recurrent.weight.data[0])

                # Calculate loss
                loss = self.criterion(outputs[-self.recall_duration:], target)
                if self.wandb_logging:
                    wandb.log({"loss": loss.item(),
                            "Train index": train_index})

                # Backward pass and optimization
                self.optimizer.zero_grad()

                loss.backward()

                if indices is not None and self.target_sparcity != 1.0:
                    layer = self.net.lif1.recurrent
                    for idx in indices["indices"]:
                        layer.weight.data[idx] = 0
                        layer.weight.grad[idx] = 0

                    self.net.lif1.recurrent = layer

                self.optimizer.step()

                if indices is not None and self.target_sparcity != 1.0:
                    layer = self.net.lif1.recurrent
                    for idx in indices["indices"]:
                        layer.weight.data[idx] = 0
                        layer.weight.grad[idx] = 0

                    self.net.lif1.recurrent = layer

                train_index+=1
            # Print loss
            if (epoch + 1) % 10 == 0 or epoch == 0:

                accuracy, val_index = self.eval(device, val_index)
                accuracies.append(accuracy)
                #print("ACCURACY",accuracy)

                temp = f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}"
                
                if dut is not None:
                    dut._log.info(temp)
                else:
                    print(temp)
               
                #wandb.log({"Train Accuracy": accuracy})
            
            # remove connections at each epoch
            if self.target_sparcity != 1.0:
                    mapping = utils.choose_conn_remove(mapping, reps=conn_reps)
                    indices = mapping.indices_to_lock
                    
                    #print("Updated weights:\n", self.net.lif1.recurrent.weight.data)
            
        if self.target_sparcity != 1.0:
            num_long_range_conns, num_short_range_conns = utils.calculate_lr_sr_conns(mapping, self.graph)
            print("lr:",num_long_range_conns,"// sr:",num_short_range_conns)
        
        final_accuracy, val_index = self.eval(device, val_index)
        accuracies.append(final_accuracy)
        print("FINAL ACCURACY",final_accuracy)
        return self.net, mapping, max(accuracies)
