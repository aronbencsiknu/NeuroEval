import torch
import snntorch as snn
import snntorch.functional as SF
from snntorch import RSynaptic
import torch.nn as nn
from snntorch import surrogate

class SpikingNet(torch.nn.Module):
    def __init__(self, opt, spike_grad=surrogate.fast_sigmoid(), learn_alpha=True, learn_beta=True, learn_treshold=True):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(opt.num_inputs, opt.num_hidden1)
        self.fc1.__setattr__("bias",None) # biological plausability
        self.lif1 = RSynaptic(alpha=0.9, beta=0.9, spike_grad=spike_grad, learn_alpha=True, learn_threshold=True, linear_features=opt.num_hidden1, reset_mechanism="subtract", reset_delay=False, all_to_all=True)
        self.lif1.recurrent.__setattr__("bias",None) # biological plausability

        self.fc2 = nn.Linear(opt.num_hidden1, opt.num_outputs)
        self.fc2.__setattr__("bias",None) # biological plausability
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        # #Set all the weights to 1
        #nn.init.constant_(self.fc1.weight, 1.0)
        #nn.init.constant_(self.fc2.weight, 1.0)

        self.num_steps = opt.num_steps

    def init_neurons():
        pass

    def forward_one_ts(self, x, spk1, syn1, mem1, mem2, time_first=True):
        #spk1, syn1, mem1 = self.lif1.init_rsynaptic()
        #mem2 = self.lif2.init_leaky()
        if not time_first:
            #test = data
            x=x.transpose(1, 0)

        ## Input layer
        cur1 = self.fc1(x)

        ### Recurrent layer
        spk1, syn1, mem1 = self.lif1(cur1, spk1, syn1, mem1)

        ### Output layer
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)

        return spk2, spk1, syn1, mem1, mem2

    def forward(self, x, time_first=True):

        spk1, syn1, mem1 = self.lif1.init_rsynaptic()
        mem2 = self.lif2.init_leaky()

        # Record the spikes from the hidden layer (if needed)
        spk1_rec = [] # not necessarily needed for inference
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        if not time_first:
            #test = data
            x=x.transpose(1, 0)

        # Print the shape of the new tensor to verify the dimensions are swapped
        #print(x.shape)
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