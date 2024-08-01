from .model import SpikingNet
from .train import Trainer
from .graph import Graph
from .mapping import Mapping
from . import utils
from .options import Variables
from .options import Specs

import torch
import math
import snntorch as snn
import hashlib

v = Variables()
s = Specs()

def generate_packets(dut):

    # -------------------------------------------------

    torch.manual_seed(42)
    # Initialize the network
    net = SpikingNet(v)

    # -------------------------------------------------

    sample_data = torch.randn(v.num_steps, v.num_inputs)
    net = utils.init_network(net, sample_data)

    indices_to_lock = {
        "indices" : ((1,2),(3,90)),
        "layers"  : ("lif1","lif1")}

    # -------------------------------------------------

    gp = Graph(v.num_steps, v.num_inputs)
    gp.export_model(net)
    gp.extract_edges()
    gp.process_graph()
    #gp.plot_graph()
    gp.log(dut)

    # -------------------------------------------------

    mapping = Mapping(net, v.num_steps, v.num_inputs, indices_to_lock)
    total_neurons = sum(mapping.mem_potential_sizes.values())
    core_capacity = max(math.ceil((total_neurons - v.num_outputs) / (v.num_cores - 1)), v.num_outputs)
    mapping.set_core_capacity(core_capacity)
    mapping.map_neurons()
    
    mapping.log(dut)

    # -------------------------------------------------

    trainer = Trainer(net,
                      mapping,
                      gp,
                      num_epochs=v.num_epochs, 
                      learning_rate=v.lr, 
                      target_frequency=v.target_fr, 
                      batch_size=v.bs, 
                      num_steps=v.num_steps)
    
    net = trainer.train(v.device, dut)

    # -------------------------------------------------

    # Dictionary to store spikes from each layer

    spike_record = {}
    hooks = []

    def reset_spike_record_and_hooks():
        global spike_record, hooks

        # Clear the spike_record dictionary
        spike_record = {}

        # Remove existing hooks if they are already registered
        if 'hooks' in globals():
            for hook in hooks:
                hook.remove()
                hooks = []

    # Function to create a hook that records spikes
    def create_spike_hook(layer_name):
        def hook(module, input, output):
            if layer_name not in spike_record:
                spike_record[layer_name] = []
            spike_record[layer_name].append(output[0].detach().cpu())
        return hook

    reset_spike_record_and_hooks()

    # Attach hooks automatically to all Leaky layers
    for name, module in net.named_modules():
        if isinstance(module, snn.Leaky) or isinstance(module, snn.RSynaptic):
            hooks.append(module.register_forward_hook(create_spike_hook(name)))

    #net, hooks = utils.attach_hooks(net)

    # -------------------------------------------------

    # Dictionary to store spikes from each layer

    spike_record = {}
    hooks = []

    # Generate random input data
    indices = [1]
    inputs = trainer.xor_inputs[indices]

    # Generate input
    inputs = trainer.generate_spike_train(inputs, v.num_steps).to(v.device)

    # Record spikes
    _, _ = net(inputs)

    # Convert spike records to tensors for easier analysis
    for layer_name in spike_record:
        spike_record[layer_name] = torch.squeeze(torch.stack(spike_record[layer_name]))

    # -------------------------------------------------

    if 'hooks' in globals():
        for hook in hooks:
            hook.remove()
            hooks = []

    # -------------------------------------------------

    routing_matrices = {}
    routing_map = {}

    for layer_name, size in mapping.mem_potential_sizes.items(): # a way to get the layer names
        # routing_matrix = torch.zeros((opt.num_steps, size))
        routing_matrix = torch.zeros((size))
        for idx in range(size):

            if layer_name in routing_matrices:
                continue

            routing_id = layer_name +"-"+ str(idx)
            source_core = mapping.neuron_to_core[routing_id]

            downstream_nodes = list(gp.graph.successors(layer_name))

            target_cores = []
            for downstream_node in downstream_nodes:
                if downstream_node != "output":
                    target_cores.extend(mapping.NIR_to_cores[downstream_node])

            # Remove skipped packets
            target_cores = utils.remove_unnecessary_packets(source_core, target_cores, mapping.buffer_map)

            # bundle packets (bundle several unicast packets into multicast)
            bundled_core_to_cores = []
            while len(target_cores) > 0:
                _, minimum = target_cores[0]
                for _, reps in target_cores:
                    if reps < minimum:
                        minimum = reps

                bcc, target_cores = utils.bundle_target_cores(target_cores, minimum)
                bundled_core_to_cores.append((bcc, minimum))

            packet_information = []

            for bcc, reps in bundled_core_to_cores:
                packet_information.append((source_core, bcc, reps))
                h = int(hashlib.shake_256(routing_id.encode()).hexdigest(2), 16)
                routing_map[h] = packet_information

                routing_matrix[idx] = h

        routing_matrices[layer_name] = routing_matrix

    packets = []

    for t in range(v.num_steps):
        packets_in_ts = []
        for layer_name, _ in mapping.mem_potential_sizes.items():

            p = utils.dot_product(routing_matrices[layer_name], 
                                        spike_record[layer_name][t], 
                                        routing_map,
                                        )
            
            packets_in_ts.extend(p)

        packets.append(packets_in_ts)

    #final_packets_list = []
    final_packets_dict = {
        s.EAST: [],
        s.NORTH: [],
        s.WEST: [],
        s.SOUTH: [],
        s.L1: []
    }
    expanded_packets_list = []

    for packet in packets:

        temp, expanded_packets = utils.repeat_and_convert_packets(packet, final_packets_dict)
        
        #final_packets_list.append(packets)
        expanded_packets_list.append(expanded_packets)

        for key in final_packets_dict:
            if key in temp:
                final_packets_dict[key].append(temp[key])

    return final_packets_dict, expanded_packets_list