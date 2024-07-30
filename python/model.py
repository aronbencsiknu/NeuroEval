import torch
import snntorch as snn
import snntorch.functional as SF
from snntorch import RSynaptic
import torch.nn as nn

class SpikingNet(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(opt.num_inputs, opt.num_hidden1)
        self.fc1.__setattr__("bias",None)
        # self.fc1.weight.data.fill_(0.01)  # Initialize all weights to 0.1
        self.lif1 = snn.RSynaptic(alpha=0.9, beta=0.9, linear_features=opt.num_hidden1, reset_mechanism="subtract", reset_delay=False, all_to_all=True)
        self.lif1.recurrent.__setattr__("bias",None)
        # self.lif1.recurrent.weight.data.fill_(0.05)  # Initialize all recurrent weights to 0.1

        self.fc2 = nn.Linear(opt.num_hidden1, opt.num_outputs)
        self.fc2.__setattr__("bias",None)
        # self.fc2.weight.data.fill_(0.05)  # Initialize all weights to 0.1
        self.lif2 = snn.Leaky(beta=0.9)

        self.num_steps = opt.num_steps

    def forward(self, x, indices_to_lock = None):

        def zero_and_lock_weights(layer, indices):
          # Step 2: Access and modify the weights at specified indices
          for idx in indices:
              layer.weight.data[idx] = 0

          # Function to zero out specific gradients
          def zero_out_grads(grad):
              for idx in indices:
                  grad[idx] = 0
              return grad

          # Step 3: Register the hook
          layer.weight.register_hook(zero_out_grads)

          return layer

        ##### Initialize hidden states at t=0 #####

        spk1, syn1, mem1 = self.lif1.init_rsynaptic()
        mem2 = self.lif2.init_leaky()

        # Record the spikes from the hidden layer (if needed)
        spk1_rec = [] # not necessarily needed for inference
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        if indices_to_lock is not None:

          # Zero out and lock the specified weights

          self.lif1.recurrent = zero_and_lock_weights(self.lif1.recurrent, indices_to_lock)

        for step in range(self.num_steps):
            ## Input layer
            cur1 = self.fc1(x[step])

            ### Recurrent layer
            spk1, syn1, mem1 = self.lif1(cur1, spk1, syn1, mem1)

            ### Output layer
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)