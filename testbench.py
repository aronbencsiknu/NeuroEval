import cocotb
from cocotb.triggers import Timer, Edge, Event

from python_files.options import Variables
from python_files.options import Specs
from python_files import generate_packets as gp

v = Variables()
s = Specs()

num_sent_messages = 0
num_recieved_messages = 0
expanded_packets = []
delays = []

global_variable_event = Event()
timestep = 30

packets = []

@cocotb.test()
async def testbench(dut):

    global expanded_packets
    global delays
    global timestep
    global packets

    #packets, expanded_packets = gp.generate_packets(dut=None)
    net, routing_matrices, routing_map, mapping, train_set, val_set = gp.snn_init(dut=None)
    packets, expanded_packets = gp.delay_experiment(net, routing_matrices, routing_map, mapping, val_set)

    #print(expanded_packets)

    #print(packets[0][10])
    #print(expanded_packets[10])
    for ts in expanded_packets:
        delays.append(dict.fromkeys(ts.keys()))

    await init(dut)

    _t = cocotb.utils.get_sim_time('ps')
    dut._log.info("Time reported = %d", _t)

    while len(expanded_packets[timestep]) == 0:
            timestep += 1

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
    string = "Packets not delivered:"+ str(len(expanded_packets[timestep-1]))
    dut._log.info(string)
    dut._log.info("TEST COMPLETED")
    dut._log.info(v.num_steps)

    for idx, key in enumerate(delays[timestep-1].keys()):
        print(delays[timestep-1][key], key)

        if idx >= 100:
            break

async def stimulus(dut, data,  input_req, ack, direction):
    global num_sent_messages
    global expanded_packets
    global timestep
    global packets
    
    index = 0
    while True:
        index += 1
        if index >= 2:
            break
        num_elements = len(packets[direction][timestep])

        counter = 0

        while counter < num_elements:
            
            if ack.value == input_req.value:

                num_sent_messages += 1
                await Timer(150, units='ps') # STABILITY
                input = int(packets[direction][timestep][counter], base=2)
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
    while True:
        await Edge(dout)
        #print("TIMESTEP",timestep)
        num_recieved_messages += 1
        message = str(dout.value)[:20]
        if message in expanded_packets[timestep]:
            # dut._log.info(expanded_packets[timestep][message])
            expanded_packets[timestep][message] -= 1

            if expanded_packets[timestep][message] == 0:
                del expanded_packets[timestep][message]
                if message in delays[timestep]:
                    start_time = float(delays[timestep][message])
                    delays[timestep][message] = float(cocotb.utils.get_sim_time('ps')) - start_time

                    if len(expanded_packets[timestep]) == 0 and timestep != v.num_steps-1:
                        timestep += 1
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
        if message in delays[timestep]:
            #print("IN DELAY")
            delays[timestep][message] = cocotb.utils.get_sim_time('ps')

async def back_ack(dut,ack, req):
    
    while True:
        await Edge(req)
        #_t = cocotb.utils.get_sim_time('ps')
        #dut._log.info("Time reported ZERO = %d", _t)
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