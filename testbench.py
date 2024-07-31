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
from python import utils 

def python_test(dut):

    # -------------------------------------------------

    torch.manual_seed(42)
    opt = Options()

    # Initialize the network
    net = SpikingNet(opt)

    # -------------------------------------------------

    sample_data = torch.randn(opt.num_steps, opt.num_inputs)
    net = utils.init_network(net, sample_data)

    indices_to_lock = {
        "indices" : ((0, 1), (90, 50)),
        "layers"  : ("lif1","lif1")}

    # -------------------------------------------------

    gp = Graph(opt.num_steps, opt.num_inputs)
    gp.export_model(net)
    gp.extract_edges()
    gp.process_graph()
    gp.plot_graph()
    gp.log(dut)

    # -------------------------------------------------

    mapping = Mapping(net, opt.num_steps, opt.num_inputs, indices_to_lock)
    total_neurons = sum(mapping.mem_potential_sizes.values())
    core_capacity = max(math.ceil((total_neurons - opt.num_outputs) / (opt.num_cores - 1)), opt.num_outputs)
    mapping.set_core_capacity(core_capacity)
    mapping.map_neurons()
    
    mapping.log(dut)

    # -------------------------------------------------

    trainer = Trainer(net, 
                      num_epochs=opt.num_epochs, 
                      learning_rate=opt.lr, 
                      target_frequency=opt.target_fr, 
                      batch_size=opt.bs, 
                      num_steps=opt.num_steps)
    
    net = trainer.train(opt.device, dut)

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
    inputs = trainer.generate_spike_train(inputs, opt.num_steps).to(opt.device)

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
                # for t in range(opt.num_steps):
                #     routing_matrix[t][idx] = h

                routing_matrix[idx] = h

        routing_matrices[layer_name] = routing_matrix

    packets = []

    for t in range(opt.num_steps):
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
        EAST: [],
        NORTH: [],
        WEST: [],
        SOUTH: [],
        L1: []
    }
    expanded_packets_list = []

    for packet in packets:

        print(len(packet))
        temp, expanded_packets = utils.repeat_and_convert_packets(packet, final_packets_dict)

        print("LEN",idx,len(expanded_packets))
        
        #final_packets_list.append(packets)
        expanded_packets_list.append(expanded_packets)

        for key in final_packets_dict:
            if key in temp:
                final_packets_dict[key].append(temp[key])

    return final_packets_dict, expanded_packets_list

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

# def generate_packets(direction):

#     direction_zero_bit_map = {
#         EAST: 4,
#         NORTH: 3,
#         WEST: 2,
#         SOUTH: 1,
#         L1: 0
#     }

#     for i in range(NUM_PACKETS_P_INJ):
#         if direction in direction_zero_bit_map:
#             zero_bit = direction_zero_bit_map[direction]
#             address_lists[direction].append(generate_address(zero_bit, valid_combinations))
#             message_lists[direction].append(generate_message())

#     return address_lists, message_lists

# def generate_message():
#     # Generate a random number in the range [0, 2^MSG_W - 1]
#     message = random.randint(0, (1 << MSG_W) - 1)
    
#     # Convert the message to a binary string, zero-padded to MSG_W bits
#     message_bits = f'{message:0{MSG_W}b}'

#     print("HELLO????? ",type(message_bits))
    
#     return message_bits

# def generate_valid_address_combinations(addr_w):
#     valid_combinations = []
#     for num_ones in range(2, addr_w + 1):  # from exactly two '1's to addr_w '1's
#         for combo in combinations(range(addr_w), num_ones):
#             bits = ['0'] * addr_w
#             for bit in combo:
#                 bits[bit] = '1'
#             valid_combinations.append(int(''.join(bits),2))
#     return valid_combinations

# valid_combinations = generate_valid_address_combinations(ADDR_W)

# def generate_address(zero_bit, valid_combinations):
#     # Select a random address from valid_combinations
#     address = valid_combinations[random.randint(0, len(valid_combinations) - 1)]
#     #address = int("11111", base=2)
#     # Convert address to binary list of bits
#     address_bits = list(f'{address:0{ADDR_W}b}')
    
#     # Set the specified bit to zero
#     address_bits[zero_bit] = '0'

#     address = ''.join(address_bits)

#     print("HELLOHELLO: ",type(address))
    
#     return address

# def concatenate_message_address(address_list, message_list, direction):
#     concatenated_list = []
#     for i in range(NUM_PACKETS_P_INJ):
#         concatenated_list.append(message_list[direction][i] + address_list[direction][i])
#     return concatenated_list


@cocotb.test()
async def replicate_data_in_e_signal(dut):

    packets, expanded_packets = python_test(dut=None)

    await init(dut)

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

    cocotb.start_soon(stimulus(dut, dut.DataInE, dut.ReqInE, dut.AckInE, packets[EAST][10]))
    cocotb.start_soon(stimulus(dut, dut.DataInN, dut.ReqInN, dut.AckInN, packets[NORTH][10]))
    cocotb.start_soon(stimulus(dut, dut.DataInW, dut.ReqInW, dut.AckInW, packets[WEST][10]))
    cocotb.start_soon(stimulus(dut, dut.DataInS, dut.ReqInS, dut.AckInS, packets[SOUTH][10]))
    cocotb.start_soon(stimulus(dut, dut.DataInL1, dut.ReqInL1, dut.AckInL1, packets[L1][10]))


    await Timer(50000000, units='ns') # Expected sim end

    dut._log.info(num_sent_messages)
    dut._log.info(num_recieved_messages)
    dut._log.info(len(expanded_packets[10]))
    dut._log.info("TEST COMPLETED")
    

async def stimulus(dut, data,  input_req, ack, packets):

    num_elements = len(packets)
    print("NUM ELEMENTS", num_elements)
    global num_sent_messages

    #limit = 5
    counter = 0

    while counter < num_elements:
        
        if ack.value == input_req.value:
            num_sent_messages += 1
            await Timer(150, units='ps') # STABILITY
            input = int(packets[counter], base=2)
            #input = 0b000010011001111
            data.value = input
            await Timer(150, units='ps') # STABILITY
            input_req.value = not input_req.value
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

    dut.SID.value = SID # L1
    dut.E_MASK.value = E_MASK
    dut.W_MASK.value = W_MASK
    dut.N_MASK.value = N_MASK
    dut.S_MASK.value = S_MASK

    await Timer(500, units='ps')