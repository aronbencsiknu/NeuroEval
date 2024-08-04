import cocotb
from cocotb.triggers import Timer, Edge

from python_files.options import Variables
from python_files.options import Specs
from python_files import generate_packets as gp

v = Variables()
s = Specs()

num_sent_messages = 0
num_recieved_messages = 0
expanded_packets = []

@cocotb.test()
async def testbench(dut):

    global expanded_packets

    packets, expanded_packets = gp.generate_packets(dut=None)

    await init(dut)

    
    timestep = 10 # temporary

    # Monitor output ports
    cocotb.start_soon(monitor_output(dut, dut.DataOutE, "EAST", timestep))
    cocotb.start_soon(monitor_output(dut, dut.DataOutN, "NORTH", timestep))
    cocotb.start_soon(monitor_output(dut, dut.DataOutW, "WEST", timestep))
    cocotb.start_soon(monitor_output(dut, dut.DataOutS, "SOUTH", timestep))
    cocotb.start_soon(monitor_output(dut, dut.DataOutL1, "LOCAL", timestep))


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
    cocotb.start_soon(stimulus(dut, dut.DataInE, dut.ReqInE, dut.AckInE, packets[s.EAST][timestep][:10]))
    cocotb.start_soon(stimulus(dut, dut.DataInN, dut.ReqInN, dut.AckInN, packets[s.NORTH][timestep][:10]))
    cocotb.start_soon(stimulus(dut, dut.DataInW, dut.ReqInW, dut.AckInW, packets[s.WEST][timestep][:10]))
    cocotb.start_soon(stimulus(dut, dut.DataInS, dut.ReqInS, dut.AckInS, packets[s.SOUTH][timestep][:10]))
    cocotb.start_soon(stimulus(dut, dut.DataInL1, dut.ReqInL1, dut.AckInL1, packets[s.L1][timestep][:10]))

    await Timer(100000, units='ns') # Expected sim end

    string = "Sent packets:"+ str(num_sent_messages)
    dut._log.info(string)
    string = "Recieved packets:"+ str(num_recieved_messages)
    dut._log.info(string)
    string = "Packets not delivered:"+ str(len(expanded_packets[timestep]))
    dut._log.info(string)
    dut._log.info("TEST COMPLETED")

async def stimulus(dut, data,  input_req, ack, packets):

    num_elements = len(packets)
    print("NUM ELEMENTS", num_elements)
    global num_sent_messages

    counter = 0

    while counter < num_elements:
        
        if ack.value == input_req.value:

            num_sent_messages += 1
            await Timer(150, units='ps') # STABILITY
            input = int(packets[counter], base=2)
            data.value = input
            await Timer(150, units='ps') # STABILITY
            input_req.value = not input_req.value
            await Timer(10, units='ps') # STABILITY
            counter += 1
        else:
            await Edge(ack)

async def monitor_output(dut,dout, name, timestep):
    global num_recieved_messages
    global expanded_packets
    while True:
        await Edge(dout)
        num_recieved_messages += 1
        string = "RECIEVED : " + name
        # dut._log.info(string)
        # dut._log.info(dout.value)
        if dout.value in expanded_packets[timestep]:
            expanded_packets[timestep].remove(dout.value)
        else:
            dut._log.info("OUT PACKET NOT FOUND")

async def monitor_input(dut, din, name):
    global num_sent_messages
    while True:
        await Edge(din)
        #num_sent_messages += 1
        string = "SENT : " + name
        # dut._log.info(string)
        # dut._log.info(din.value)
        # print()

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