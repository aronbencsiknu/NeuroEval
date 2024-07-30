import torch
import snntorch as snn  # Ensure this module is correctly imported

class Mapping:
    def __init__(self, net, num_steps, num_inputs, core_capacity, indices_to_lock):
        self.num_steps = num_steps
        self.num_inputs = num_inputs
        self.core_capacity = core_capacity
        self.net = net

        self.mem_potential_sizes = self.get_membrane_potential_sizes()
        self.core_allocation, self.NIR_to_cores, self.neuron_to_core = self.allocate_neurons_to_cores()
        self.buffer_map = self.map_buffers(indices_to_lock)
    
    def get_membrane_potential_sizes(self):
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
    
    def allocate_neurons_to_cores(self):
        core_allocation = {}
        NIR_to_cores = {}
        neuron_to_core = {}

        total_neurons = sum(self.mem_potential_sizes.values())

        core_id = 0
        core_start_index = 0
        current_core_neurons = 0

        layer_names = list(self.mem_potential_sizes.keys())
        last_layer_name = layer_names[-1]

        for layer_name, num_neurons in self.mem_potential_sizes.items():
            layer_start_index = core_start_index

            if layer_name == last_layer_name:
                if num_neurons > self.core_capacity:
                    raise Exception("Output layer does not fit in one core!")

                # Ensure the last layer is in the same core
                core_id += 1
                core_start_index = 0
                current_core_neurons = 0
                layer_start_index = core_start_index
                layer_end_index = layer_start_index + num_neurons - 1
                core_allocation[layer_name] = [(core_id, layer_start_index, layer_end_index)]
                NIR_to_cores[layer_name] = [(core_id, layer_end_index - layer_start_index)]
                for neuron_id in range(layer_start_index, layer_end_index + 1):
                    neuron_to_core[layer_name + "-" + str(neuron_id)] = core_id
                break

            while num_neurons > 0:
                available_space = self.core_capacity - current_core_neurons
                neurons_to_allocate = min(num_neurons, available_space)

                layer_end_index = layer_start_index + neurons_to_allocate - 1

                if layer_name not in core_allocation:
                    core_allocation[layer_name] = []
                    NIR_to_cores[layer_name] = []

                core_allocation[layer_name].append((core_id, layer_start_index, layer_end_index))
                NIR_to_cores[layer_name].append((core_id, layer_end_index - layer_start_index))

                for neuron_id in range(layer_start_index, layer_end_index + 1):
                    neuron_to_core[layer_name + "-" + str(neuron_id)] = core_id

                current_core_neurons += neurons_to_allocate
                layer_start_index += neurons_to_allocate
                num_neurons -= neurons_to_allocate

                if current_core_neurons == self.core_capacity:
                    core_id += 1
                    core_start_index = 0
                    current_core_neurons = 0
                else:
                    core_start_index = layer_start_index

        return core_allocation, NIR_to_cores, neuron_to_core
    
    def map_buffers(self, indices_to_lock):
        mapped_buffer = {}
        for indices in indices_to_lock['indices']:
            temp = ""
            for idx, layer_name in enumerate(indices_to_lock['layers']):
                temp += str(self.neuron_to_core[layer_name + "-" + str(indices[idx])])

            if temp not in mapped_buffer:
                mapped_buffer[temp] = 1
            else:
                mapped_buffer[temp] += 1

        return mapped_buffer

# Example usage:
# num_steps = 10
# num_inputs = 5
# core_capacity = 100  # Example core capacity
# net = ...  # Define or load your network model here
# indices_to_lock = ...  # Define indices to lock

# mapping = Mapping(num_steps, num_inputs, core_capacity)
# mapping.set_network(net)
# mem_potential_sizes = mapping.get_membrane_potential_sizes()
# for layer_name, size in mem_potential_sizes.items():
#     print(f"Layer: {layer_name}, Number of neurons: {size}")

# core_allocation, NIR_to_cores, neuron_to_core = mapping.allocate_neurons_to_cores(mem_potential_sizes)
# print(core_allocation)
# print(NIR_to_cores)
# print(neuron_to_core)

# for layer_name, allocations in core_allocation.items():
#     print(f"Layer: {layer_name}")
#     for core_id, start_idx, end_idx in allocations:
#         print(f"  Core {core_id}: start index = {start_idx}, end index = {end_idx}")

# buffer_map = mapping.map_buffers(indices_to_lock, neuron_to_core)
# print(buffer_map)
