import torch
import snntorch as snn  # Ensure this module is correctly imported
import random
from .options import Variables

v = Variables()

class Mapping:
    def __init__(self, net, num_steps, num_inputs):
        self.num_steps = num_steps
        self.num_inputs = num_inputs
        self.core_capacity = None
        self.net = net

        self.mem_potential_sizes = self._get_membrane_potential_sizes()
        self.buffer_map = None
        self.indices_to_lock = None
    
    def _get_membrane_potential_sizes(self):
        if self.net is None:
            raise ValueError("Network model has not been set. Please call set_network first.")
        
        sizes = {}
        for name, module in self.net.named_modules():
            if isinstance(module, snn.Synaptic):
                _, mem = module.init_leaky()
                sizes[name] = mem.size()[0]

            elif isinstance(module, snn.Leaky):
                mem = module.init_leaky()
                sizes[name] = mem.size()[0]

            elif isinstance(module, snn.RSynaptic):
                sizes[name] = module.linear_features

        return sizes
    
    def map_neurons(self):
        self.neuron_to_core = self._allocate_neurons_to_cores()

    def set_core_capacity(self, cc):
        self.core_capacity = cc

    def log(self, dut=None):

        print("\n----- MAPPING -----\n")

        for layer_name, size in self.mem_potential_sizes.items():
            temp = f"Layer: {layer_name}, Number of neurons: {size}"
            if dut is not None:
                    dut._log.info(temp)
            else:
                print(temp)

        # print("CORE ALLOCATION:",self.core_allocation)
        # print("NIR TO CORES:",self.NIR_to_cores)
        # print("BUFFER MAP:",self.buffer_map)

        # print("CORE CAPACITY", self.core_capacity)
    
    def _allocate_neurons_to_cores(self):
        print("MAPPING")
        neuron_to_core = {}

        core_av = {i: [0, False] for i in range(v.num_cores-1)}
        # Return a random choice from these keys
        core_id = 0

        layer_names = list(self.mem_potential_sizes.keys())
        last_layer_name = layer_names[-1]

        # iterate through each layer
        neuron_counter = 0
        for layer_name, num_neurons in self.mem_potential_sizes.items():

            if layer_name == last_layer_name:
                if num_neurons > self.core_capacity:
                    raise Exception("Output layer does not fit in one core!")

                for neuron_id in range(0, num_neurons):
                    core_id = v.num_cores-1
                    neuron_to_core[layer_name + "-" + str(neuron_id)] = core_id
                    #core_av[core_id] += 1
                break
            else:  
                for neuron_id in range(0, num_neurons):
                    neuron_to_core[layer_name + "-" + str(neuron_id)] = core_id
                    neuron_counter+=1
                    switch = random.random() < 1.0
                    if core_av[core_id][0] >= self.core_capacity or (switch and not core_av[core_id][1]):
                        core_av[core_id][1] = True

                        available_cores = [core_id for core_id, value in core_av.items() if value[0] < self.core_capacity]
                        core_id = random.choice(available_cores)
                        core_av[core_id][0] += 1
                    else:
                        core_av[core_id][0] += 1
        print(self.core_capacity)
        print("COUNTER", neuron_counter)
        print(core_av)
        return neuron_to_core
    
    def map_buffers(self, indices_to_lock=None):

        if indices_to_lock is not None:
            self.indices_to_lock = indices_to_lock

        mapped_buffer = {}
        for indices in self.indices_to_lock['indices']:
            temp = ""
            #for idx, layer_name in enumerate(indices_to_lock['layers']):
                #if idx == 0:
            temp += str(self.indices_to_lock['layers'][0])+"-"+str(indices[0]) +"-"
            temp += str(self.neuron_to_core[str(self.indices_to_lock['layers'][1]) + "-" + str(indices[1])])

            if temp not in mapped_buffer:
                mapped_buffer[temp] = 1
            else:
                mapped_buffer[temp] += 1

        #return mapped_buffer
        self.buffer_map = mapped_buffer
