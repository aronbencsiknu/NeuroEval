# This file is public domain, it can be freely copied without restrictions.
# SPDX-License-Identifier: CC0-1.0

import cocotb
from cocotb.triggers import Timer, Edge
import random
from itertools import combinations
import torch
import snntorch as snn
from snntorch import spikeplot as splt
import matplotlib.pyplot as plt
import torch
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt
import torch
import hashlib
import math

from python.model import SpikingNet
from python.train import Trainer
from python.options import Options
from python.graph import Graph
from python.mapping import Mapping

def python_test():

    # -------------------------------------------------

    opt = Options()
    # Initialize the network
    net = SpikingNet(opt)

    # -------------------------------------------------

    torch.manual_seed(42)
    sample_data = torch.randn(opt.num_steps, opt.num_inputs)

    indices_to_lock = {
        "indices" : ((0, 1), (90, 50)),
        "layers"  : ("lif1","lif1")}
    
    # init inference
    try:
        prediction, _ = net(sample_data, indices_to_lock["indices"])
    except:
        prediction, _ = net(sample_data, indices_to_lock["indices"])

    print("Updated weights:\n", net.lif1.recurrent.weight.data)

    # -------------------------------------------------

    gp = Graph(opt.num_steps, opt.num_inputs)
    gp.export_model(net)
    gp.extract_edges()
    gp.process_graph()
    gp.plot_graph()

    # -------------------------------------------------

    mapping = Mapping(net, opt.num_steps, opt.num_inputs, indices_to_lock)

    total_neurons = sum(mapping.mem_potential_sizes.values())

    core_capacity = max(math.ceil((total_neurons - opt.num_outputs) / (opt.num_cores - 1)), opt.num_outputs)

    mapping.set_core_capacity(core_capacity)

    mapping.map_neurons()
    
    #mem_potential_sizes = mapping.get_membrane_potential_sizes()
    for layer_name, size in mapping.mem_potential_sizes.items():
        print(f"Layer: {layer_name}, Number of neurons: {size}")

    for layer_name, allocations in mapping.core_allocation.items():
        print(f"Layer: {layer_name}")
        for core_id, start_idx, end_idx in allocations:
            print(f"  Core {core_id}: start index = {start_idx}, end index = {end_idx}")

    print(mapping.core_allocation)
    print(mapping.NIR_to_cores)
    print(mapping.neuron_to_core)
    print(mapping.buffer_map)

    # -------------------------------------------------

    trainer = Trainer(net, 
                      num_epochs=opt.num_epochs, 
                      learning_rate=opt.lr, 
                      target_frequency=opt.target_fr, 
                      batch_size=opt.bs, 
                      num_steps=opt.num_steps)
    
    net = trainer.train(opt.device)

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

    # -------------------------------------------------

    # Dictionary to store spikes from each layer

    spike_record = {}
    hooks = []

    # Generate random input data
    indices = [1]
    inputs = trainer.xor_inputs[indices]

    # Generate spike trains
    inputs = trainer.generate_spike_train(inputs, opt.num_steps).to(opt.device)

    output, _ = net(inputs)

    # Convert spike records to tensors for easier analysis
    for layer_name in spike_record:
        spike_record[layer_name] = torch.squeeze(torch.stack(spike_record[layer_name]))

    print(spike_record['lif1'].shape)

    # # Plot spike rasters for all layers
    # num_layers = len(spike_record)
    # fig, ax = plt.subplots(num_layers, 1, figsize=(10, 4 * num_layers))


    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)

    print(output.shape)

    #  s: size of scatter points; c: color of scatter points
    splt.raster(output.squeeze(1), ax, s=1.5, c="black")
    plt.title("Output Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()

    # -------------------------------------------------

    if 'hooks' in globals():
        for hook in hooks:
            hook.remove()
            hooks = []

    # -------------------------------------------------

    def bundle_target_cores(target_cores, min_reps):
        res = []
        new_target_cores = []
        for target_core, reps in target_cores:
            res.append(target_core)
            if reps - min_reps > 0:
                new_target_cores.append((target_core, reps - min_reps))

        return res, new_target_cores

    def remove_unnecessary_packets(source_core, target_cores, buffer_map):
        new_target_cores = []
        for target_core, reps in target_cores:
            if source_core == target_core:
                continue
            if str(source_core)+str(target_core) in buffer_map:
                new_target_cores.append((target_core, reps - int(buffer_map[str(source_core)+str(target_core)])))
            else:
                new_target_cores.append((target_core, reps))

        return new_target_cores

    routing_matrices = {}
    routing_map = {}

    for layer_name, size in mapping.mem_potential_sizes.items(): # a way to get the layer names
        routing_matrix = torch.zeros((opt.num_steps, size))
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
            target_cores = remove_unnecessary_packets(source_core, target_cores, mapping.buffer_map)

            # bundle packets (bundle several unicast packets into multicast)
            bundled_core_to_cores = []
            while len(target_cores) > 0:
                _, minimum = target_cores[0]
                for _, reps in target_cores:
                    if reps < minimum:
                        minimum = reps

                bcc, target_cores = bundle_target_cores(target_cores, minimum)
                bundled_core_to_cores.append((bcc, minimum))

            packet_information = []
            for bcc, reps in bundled_core_to_cores:
                packet_information.append((source_core, bcc, reps))
                h = int(hashlib.shake_256(routing_id.encode()).hexdigest(2), 16)
                routing_map[h] = packet_information
                for t in range(opt.num_steps):
                    routing_matrix[t][idx] = h

        routing_matrices[layer_name] = routing_matrix

    print(routing_matrices['lif1'])
    print(routing_map)

    exp = torch.mul(routing_matrices['lif1'], spike_record['lif1'])

    temp = exp[1]
    non_zero_values = temp[temp != 0]

    packets = []
    for hashes in non_zero_values:
        packets.append(routing_map[int(hashes)])

    print(packets)

ADDR_W = 5
MSG_W = 10
EAST, NORTH, WEST, SOUTH, L1 = range(5)
NUM_PACKETS_P_INJ = 20

num_sent_messages = 0
num_recieved_messages = 0

SID = 0b00001
E_MASK = 0b10000
N_MASK = 0b01000
W_MASK = 0b00100
S_MASK = 0b00010

packetCounterE = 0
packetCounterN = 0
packetCounterW = 0
packetCounterS = 0
packetCounterL1 = 0

counterRecieved = 0

address_lists = {
        EAST: [],
        NORTH: [],
        WEST: [],
        SOUTH: [],
        L1: []
    }
    
message_lists = {
        EAST: [],
        NORTH: [],
        WEST: [],
        SOUTH: [],
        L1: []
    }

expanded_packets = []

def generate_packets(direction):

    direction_zero_bit_map = {
        EAST: 4,
        NORTH: 3,
        WEST: 2,
        SOUTH: 1,
        L1: 0
    }

    for i in range(NUM_PACKETS_P_INJ):
        if direction in direction_zero_bit_map:
            zero_bit = direction_zero_bit_map[direction]
            address_lists[direction].append(generate_address(zero_bit, valid_combinations))
            message_lists[direction].append(generate_message())

    return address_lists, message_lists

def generate_message():
    # Generate a random number in the range [0, 2^MSG_W - 1]
    message = random.randint(0, (1 << MSG_W) - 1)
    
    # Convert the message to a binary string, zero-padded to MSG_W bits
    message_bits = f'{message:0{MSG_W}b}'
    
    return message_bits

def generate_valid_address_combinations(addr_w):
    valid_combinations = []
    for num_ones in range(2, addr_w + 1):  # from exactly two '1's to addr_w '1's
        for combo in combinations(range(addr_w), num_ones):
            bits = ['0'] * addr_w
            for bit in combo:
                bits[bit] = '1'
            valid_combinations.append(int(''.join(bits),2))
    return valid_combinations

valid_combinations = generate_valid_address_combinations(ADDR_W)

def generate_address(zero_bit, valid_combinations):
    # Select a random address from valid_combinations
    address = valid_combinations[random.randint(0, len(valid_combinations) - 1)]
    #address = int("11111", base=2)
    # Convert address to binary list of bits
    address_bits = list(f'{address:0{ADDR_W}b}')
    
    # Set the specified bit to zero
    address_bits[zero_bit] = '0'

    address = ''.join(address_bits)
    
    return address

def concatenate_message_address(address_list, message_list, direction):
    concatenated_list = []
    for i in range(NUM_PACKETS_P_INJ):
        concatenated_list.append(message_list[direction][i] + address_list[direction][i])
    return concatenated_list


@cocotb.test()
async def replicate_data_in_e_signal(dut):

    python_test()

    await init(dut)

    generate_packets(EAST)
    generate_packets(NORTH)
    generate_packets(WEST)
    generate_packets(SOUTH)
    generate_packets(L1)

    expanded_packets.extend(concatenate_message_address(address_lists, message_lists, EAST))
    expanded_packets.extend(concatenate_message_address(address_lists, message_lists, NORTH))
    expanded_packets.extend(concatenate_message_address(address_lists, message_lists, WEST))
    expanded_packets.extend(concatenate_message_address(address_lists, message_lists, SOUTH))
    expanded_packets.extend(concatenate_message_address(address_lists, message_lists, L1))

    # Monitor output ports
    cocotb.start_soon(monitor_output(dut, dut.DataOutE, "EAST"))
    cocotb.start_soon(monitor_output(dut, dut.DataOutN, "NORTH"))
    cocotb.start_soon(monitor_output(dut, dut.DataOutW, "WEST"))
    cocotb.start_soon(monitor_output(dut, dut.DataOutS, "SOUTH"))
    cocotb.start_soon(monitor_output(dut, dut.DataOutL1, "LOCAL"))

    # Monitor input ports
    cocotb.start_soon(monitor_input(dut, dut.DataInE, "EAST"))
    cocotb.start_soon(monitor_input(dut, dut.DataInN, "NORTH"))
    cocotb.start_soon(monitor_input(dut, dut.DataInW, "WEST"))
    cocotb.start_soon(monitor_input(dut, dut.DataInS, "SOUTH"))
    cocotb.start_soon(monitor_input(dut, dut.DataInL1, "LOCAL"))

    # Back Acknowledgement
    cocotb.start_soon(back_ack(dut, dut.AckOutE, dut.ReqOutE))
    cocotb.start_soon(back_ack(dut, dut.AckOutN, dut.ReqOutN))
    cocotb.start_soon(back_ack(dut, dut.AckOutW, dut.ReqOutW))
    cocotb.start_soon(back_ack(dut, dut.AckOutS, dut.ReqOutS))
    cocotb.start_soon(back_ack(dut, dut.AckOutL1, dut.ReqOutL1))

    # Stimuli
    cocotb.start_soon(stimulus(dut, dut.DataInE, dut.ReqInE, dut.AckInE, address_lists[EAST], message_lists[EAST]))
    cocotb.start_soon(stimulus(dut, dut.DataInN, dut.ReqInN, dut.AckInN, address_lists[NORTH], message_lists[NORTH]))
    cocotb.start_soon(stimulus(dut, dut.DataInW, dut.ReqInW, dut.AckInW, address_lists[WEST], message_lists[WEST]))
    cocotb.start_soon(stimulus(dut, dut.DataInS, dut.ReqInS, dut.AckInS, address_lists[SOUTH], message_lists[SOUTH]))
    cocotb.start_soon(stimulus(dut, dut.DataInL1, dut.ReqInL1, dut.AckInL1, address_lists[L1], message_lists[L1]))

    await Timer(80000, units='ps') # Expected sim end

    dut._log.info(num_sent_messages)
    dut._log.info(num_recieved_messages)
    dut._log.info("TEST COMPLETED")

async def stimulus(dut, data,  input_req, ack, address_list, data_list):

    num_elements = len(address_list)
    print("NUM ELEMENTS", num_elements)
    global num_sent_messages

    #limit = 5
    counter = 0

    while counter < num_elements:
        
        if ack.value == input_req.value:
            num_sent_messages += 1
            await Timer(150, units='ps') # STABILITY
            input = int(data_list[counter] + address_list[counter], base=2)
            #input = 0b000010011001111
            data.value = input
            await Timer(150, units='ps') # STABILITY
            input_req.value =  not input_req.value
            await Timer(10, units='ps') # STABILITY
            counter += 1
        else:
            await Edge(ack)

async def monitor_output(dut,dout, name):
    global num_recieved_messages
    while True:
        await Edge(dout)
        num_recieved_messages += 1
        string = "RECIEVED : " + name
        dut._log.info(string)
        dut._log.info(dout.value)
        print()

async def monitor_input(dut, din, name):
    global num_sent_messages
    while True:
        await Edge(din)
        #num_sent_messages += 1
        string = "SENT : " + name
        dut._log.info(string)
        dut._log.info(din.value)
        print()

async def back_ack(dut,ack, req):
    
    while True:
        await Edge(req)
        await Timer(10, units='ps') # STABILITY
        ack.value = req.value

async def init(dut):
    await Timer(500, units='ps')
    # Initialize the signal
    dut.rst.value = 0
    dut.AckOutE.value = 0
    dut.AckOutN.value = 0
    dut.AckOutW.value = 0
    dut.AckOutS.value = 0
    dut.AckOutL1.value = 0
    await Timer(500, units='ps')
    dut.rst.value = 1
    dut.ReqInE.value = 0
    dut.ReqInN.value = 0
    dut.ReqInW.value = 0
    dut.ReqInS.value = 0
    dut.ReqInL1.value = 0

    dut.DataInE.value = 0b000000000000000
    dut.DataInN.value = 0b000000000000000
    dut.DataInW.value = 0b000000000000000
    dut.DataInS.value = 0b000000000000000
    dut.DataInL1.value = 0b000000000000000

    dut.SID.value = SID
    dut.E_MASK.value = E_MASK
    dut.W_MASK.value = W_MASK
    dut.N_MASK.value = N_MASK
    dut.S_MASK.value = S_MASK


    await Timer(150, units='ps')