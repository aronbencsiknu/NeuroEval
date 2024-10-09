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
        #self.fc1.__setattr__("bias",None) # biological plausability

        self.lif1 = RSynaptic(alpha=0.9, beta=0.9, spike_grad=spike_grad, learn_alpha=True, learn_threshold=True, linear_features=opt.num_hidden1, reset_mechanism="subtract", reset_delay=False, all_to_all=True)
        #self.lif1.recurrent.__setattr__("bias",None) # biological plausability

        self.fc2 = nn.Linear(opt.num_hidden1, opt.num_hidden1)
        #self.fc2.__setattr__("bias",None) # biological plausability

        self.lif2 = RSynaptic(alpha=0.9, beta=0.9, spike_grad=spike_grad, learn_alpha=True, learn_threshold=True, linear_features=opt.num_hidden1, reset_mechanism="subtract", reset_delay=False, all_to_all=True)
        #self.lif2.recurrent.__setattr__("bias",None) # biological plausability

        self.fc3 = nn.Linear(opt.num_hidden1, opt.num_hidden1)
        #self.fc3.__setattr__("bias",None) # biological plausability

        self.lif3 = RSynaptic(alpha=0.9, beta=0.9, spike_grad=spike_grad, learn_alpha=True, learn_threshold=True, linear_features=opt.num_hidden1, reset_mechanism="subtract", reset_delay=False, all_to_all=True)
        #self.lif3.recurrent.__setattr__("bias",None) # biological plausability

        self.fc4 = nn.Linear(opt.num_hidden1, opt.num_hidden1)
        #self.fc4.__setattr__("bias",None) # biological plausability
        self.lif4 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        self.fc5 = nn.Linear(opt.num_hidden1, opt.num_hidden1)
        #self.fc4.__setattr__("bias",None) # biological plausability
        self.lif5 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        self.fc6 = nn.Linear(opt.num_hidden1, opt.num_outputs)
        #self.fc4.__setattr__("bias",None) # biological plausability
        self.lif6 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        self.num_steps = opt.num_steps

    def init_neurons():
        pass

    def forward_one_ts(self, x, spk1, syn1, mem1, mem2, cur_sub=None, cur_add=None, time_first=True):

        if not time_first:
            #test = data
            x=x.transpose(1, 0)
        curr_sub_rec = []
        curr_add_rec = []
        
        curr_sub_fc = []
        curr_add_fc = []
        if cur_sub is not None:
            for element in cur_sub:
                if element[2] > 99:
                    curr_sub_fc.append(element)
                    pass
                else:
                    curr_sub_rec.append(element)
                    pass

        if cur_add is not None:
            for element in cur_add:
                if element[2] > 99:
                    curr_add_fc.append(element)
                    pass
                else:
                    curr_add_rec.append(element)
                    pass

        ## Input layer
        cur1 = self.fc1(x)

        ### Recurrent layer
        spk1, syn1, mem1 = self.lif1(cur1, spk1, syn1, mem1)

        for element in curr_sub_rec:
            multiplier = element[0]
            w_idx = (element[2], element[1])
            cur_idx = element[2]

            weight = self.lif1.recurrent.weight.data[w_idx].item()

            syn1[cur_idx] = syn1[cur_idx] - weight*multiplier

        for element in curr_add_rec:
            multiplier = element[0]
            w_idx = (element[2], element[1])
            cur_idx = element[2]

            weight = self.lif1.recurrent.weight.data[w_idx].item()

            syn1[cur_idx] = syn1[cur_idx] + weight*multiplier

        ### Output layer
        cur2 = self.fc2(spk1)

        for element in curr_sub_fc:
            multiplier = element[0]
            w_idx = (element[2]-100, element[1])
            cur_idx = element[2]-100
            #print("WEIGHT DIMS", self.fc2.weight.data.shape)
            weight = self.fc2.weight.data[w_idx].item()

            cur2[cur_idx] = cur2[cur_idx] - weight*multiplier

        for element in curr_add_fc:
            multiplier = element[0]
            w_idx = (element[2]-100, element[1])
            cur_idx = element[2]-100

            weight = self.fc2.weight.data[w_idx].item()

            cur2[cur_idx] = cur2[cur_idx] + weight*multiplier

        spk2, mem2 = self.lif2(cur2, mem2)

        return spk2, spk1, syn1, mem1, mem2

    def forward(self, x, time_first=True):

        spk1, syn1, mem1 = self.lif1.init_rsynaptic()
        spk2, syn2, mem2 = self.lif2.init_rsynaptic()
        spk3, syn3, mem3 = self.lif3.init_rsynaptic()
        mem4 = self.lif4.init_leaky()
        mem5 = self.lif5.init_leaky()
        mem6 = self.lif6.init_leaky()

        # Record the spikes from the hidden layer (if needed)
        spk1_rec = [] # not necessarily needed for inference
        # Record the final layer
        spk_out_rec = []
        mem_out_rec = []

        if not time_first:
            #test = data
            x=x.transpose(1, 0)

        # Print the shape of the new tensor to verify the dimensions are swapped
        #print(x.shape)
        for step in range(self.num_steps):
            ## Input layer
            cur1 = self.fc1(x[step])

            ### Recurrent layers
            spk1, syn1, mem1 = self.lif1(cur1, spk1, syn1, mem1)

            cur2 = self.fc2(spk1)
            
            spk2, syn2, mem2 = self.lif2(cur2, spk2, syn2, mem2)

            cur3 = self.fc3(spk2)
            
            spk3, syn3, mem3 = self.lif3(cur3, spk3, syn3, mem3)

            ### Output layers
            cur4 = self.fc4(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            cur5 = self.fc5(spk4)
            spk5, mem5 = self.lif5(cur5, mem5)


            cur6 = self.fc6(spk5)
            spk6, mem6 = self.lif6(cur6, mem6)


            spk_out_rec.append(spk6)
            mem_out_rec.append(mem6)

        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)