import cocotb
from cocotb.triggers import Timer, Edge, Event
import random
import numpy as np

from python_files.options import Variables
from python_files.options import Specs
from python_files import generate_packets as gp

v = Variables()
s = Specs()

num_sent_messages = 0
num_recieved_messages = 0
expanded_packets = []
delays = []
fixed_ts_indices = []

global_variable_event = Event()
ts_index = 0
timestep = 0
sample = 0
ran_timesteps = []
start_time = None

packets = []

@cocotb.test()
async def testbench(dut):

    global expanded_packets
    global delays
    global timestep
    global packets
    global fixed_ts_indices

    random_sample_indices = []
    random_ts_indices = []

    net, routing_matrices, routing_map, mapping, _, val_set, final_accuracy = gp.snn_init(dut=None)

    random_sample_indices = random.sample(range(len(val_set)), 20)
    fixed_ts_indices = np.linspace(30, v.num_steps-1, 5).round().astype(int)
    print("NUM STEPS",v.num_steps)
    timestep = fixed_ts_indices[0]
    
    #print("LEN PAC", len(pac))
    for idx in random_sample_indices:
        pac, exp = gp.delay_experiment(net, routing_matrices, routing_map, mapping, val_set, idx)
        
        packets.append(pac)
        expanded_packets.append(exp)
    print("LEN OF FIRST SAMPLE",len(packets[0]))
    for idx in range(len(expanded_packets)):
        temp = []
        for ts in expanded_packets[idx]:
            #delays.append(dict.fromkeys(ts.keys()))
            temp.append(dict.fromkeys(ts.keys()))
        delays.append(temp)

    await init(dut)

    _t = cocotb.utils.get_sim_time('ps')
    dut._log.info("Time reported = %d", _t)

    # skip all timesteps with 0 packets
    # while len(expanded_packets[timestep]) == 0:
    #         timestep += 1
    ran_timesteps.append(timestep)

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

    # provide packets
    cocotb.start_soon(stimulus(dut, dut.DataInE, dut.ReqInE, dut.AckInE, s.EAST))
    cocotb.start_soon(stimulus(dut, dut.DataInN, dut.ReqInN, dut.AckInN, s.NORTH))
    cocotb.start_soon(stimulus(dut, dut.DataInW, dut.ReqInW, dut.AckInW, s.WEST))
    cocotb.start_soon(stimulus(dut, dut.DataInS, dut.ReqInS, dut.AckInS, s.SOUTH))
    cocotb.start_soon(stimulus(dut, dut.DataInL1, dut.ReqInL1, dut.AckInL1, s.L1))


    #cocotb.start_soon(check_packet_transit(dut, timestep))

    await Timer(100000, units='ns') # Expected sim end

    string = "Sent packets:"+ str(num_sent_messages)
    dut._log.info(string)
    string = "Recieved packets:"+ str(num_recieved_messages)
    dut._log.info(string)
    # string = "Packets not delivered:"+ str(len(expanded_packets[timestep-1]))
    # dut._log.info(string)
    dut._log.info("TEST COMPLETED")
    #dut._log.info(v.num_steps)

    avg_end_to_end = 0
    avg_noc = 0
    counter = 0
    ran_timesteps.pop()
    for ts in ran_timesteps:
        for idx, key in enumerate(delays[s][ts].keys()):
            #print(delays[timestep-1][key], key)
            counter += 1
            #print(delays[timestep-1][key], key)
            avg_end_to_end += delays[s][ts][key][0] # get ETE delay
            avg_noc += delays[s][ts][key][1] # get NoC delay

    avg_end_to_end = avg_end_to_end / counter
    avg_noc = avg_noc / counter
    print("AVERAGE DELAY ETE", avg_end_to_end)
    print("AVERAGE DELAY NOC", avg_noc)
    print("FINAL ACCURACY", final_accuracy)

async def stimulus(dut, data,  input_req, ack, direction):
    global num_sent_messages
    global expanded_packets
    global timestep
    global packets
    global start_time
    global ts_index
    
    index = 0
    while True:
        index += 1
        if ts_index >= len(fixed_ts_indices):
            print("BREAK")
            print(len(fixed_ts_indices))
            break

        print(timestep)
        num_elements = len(packets[sample][direction][timestep])

        counter = 0

        while counter < num_elements:
            
            if ack.value == input_req.value:
                if start_time is None:
                    start_time = cocotb.utils.get_sim_time('ps')

                num_sent_messages += 1
                await Timer(150, units='ps') # STABILITY
                input = int(packets[sample][direction][timestep][counter], base=2)
                data.value = input
                await Timer(150, units='ps') # STABILITY
                input_req.value = not input_req.value
                await Timer(10, units='ps') # STABILITY
                counter += 1
            else:
                await Edge(ack)

        await global_variable_event.wait()

async def monitor_output(dut,dout, name):
    global num_recieved_messages
    global expanded_packets
    global timestep
    global delays
    global start_time
    global ts_index

    while True:
        await Edge(dout)
        num_recieved_messages += 1
        message = str(dout.value)[:20]
        if message in expanded_packets[sample][timestep]:
            expanded_packets[sample][timestep][message] -= 1

            if expanded_packets[sample][timestep][message] == 0:
                del expanded_packets[sample][timestep][message]
                if message in delays[sample][timestep]:
                    inj_time = float(delays[sample][timestep][message])
                    current_time = float(cocotb.utils.get_sim_time('ps'))
                    delays[sample][timestep][message] = (current_time - start_time, current_time - inj_time)

                    if len(expanded_packets[sample][timestep]) == 0 and timestep != v.num_steps-1:
                        print("HERE ############################x")
                        ts_index += 1
                        timestep = fixed_ts_indices[ts_index]
                        ran_timesteps.append(timestep)
                        start_time = None
                        global_variable_event.set()  # Unblock the stimulus loop
                        global_variable_event.clear()  # Reset the event for future use
                    
        else:
            dut._log.info("OUT PACKET NOT FOUND")

async def monitor_input(dut, din, name):
    global num_sent_messages
    global timestep
    global delays
    while True:
        await Edge(din)
        message = str(din.value)[:20]
        if message in delays[sample][timestep]:
            pass
            #print("IN DELAY")
            delays[sample][timestep][message] = cocotb.utils.get_sim_time('ps')

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

    dut.SID.value = s.SID # L1
    dut.E_MASK.value = s.E_MASK
    dut.W_MASK.value = s.W_MASK
    dut.N_MASK.value = s.N_MASK
    dut.S_MASK.value = s.S_MASK

    await Timer(500, units='ps')