# This file is public domain, it can be freely copied without restrictions.
# SPDX-License-Identifier: CC0-1.0

import cocotb
from cocotb.triggers import FallingEdge, Timer, Edge
import random
from itertools import combinations

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

