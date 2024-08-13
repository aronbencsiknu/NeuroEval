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
max_dut_len = 20

@cocotb.test()
async def testbench(dut):

    global expanded_packets

    packets, expanded_packets = gp.generate_packets(dut=None)

    await Edge(dut.init_done)
    
    timestep = 0 # temporary

    def return_chunk(packets, st_index):
         length = len(packets)

         if st_index >=length - 1:
              return None, None, True

         if st_index + max_dut_len > length:
              
              chunk = [int(element, base=2) for element in packets[st_index:length]] + [0]*(max_dut_len-(length - st_index))
              return chunk, length - st_index, False
         
         chunk = [int(element, base=2) for element in packets[st_index:st_index + max_dut_len]]
         return chunk, max_dut_len, False

    for i in range(20,25):

 
        if not (len(packets[s.EAST][i]) == 0 and 
                len(packets[s.NORTH][i]) == 0 and 
                len(packets[s.WEST][i]) == 0 and 
                len(packets[s.SOUTH][i]) == 0 and
                len(packets[s.L1][i]) == 0):

                temp = "HELLO"+ str(i)
                # dut._log.info("--------------------------------------")
                # dut._log.info(temp)
                # dut._log.info("--------------------------------------")
                start_index = 0
                while True:
                    
                    # no append to but just send lol
                    chunk_e, length_e, finished_e = return_chunk(packets[s.EAST][i], start_index)
                    chunk_n, length_n, finished_n = return_chunk(packets[s.NORTH][i], start_index)
                    chunk_w, length_w, finished_w = return_chunk(packets[s.WEST][i], start_index)
                    chunk_s, length_s, finished_s = return_chunk(packets[s.SOUTH][i], start_index)
                    chunk_l1, length_l1, finished_l1 = return_chunk(packets[s.L1][i], start_index)
                    
                    if (finished_e and
                        finished_n and 
                        finished_w and 
                        finished_s and 
                        finished_l1):
                        
                        break
                    dut.packetCounterE.value = 0
                    dut.packetCounterN.value = 0
                    dut.packetCounterW.value = 0
                    dut.packetCounterS.value = 0
                    dut.packetCounterL1.value = 0

                    if not finished_e:
                        dut.packetListE.value = chunk_e
                        dut.packetLimitE.value = length_e
                    if not finished_n:
                        dut.packetListN.value = chunk_n
                        dut.packetLimitN.value = length_n
                    if not finished_w:
                        dut.packetListW.value = chunk_w
                        dut.packetLimitW.value = length_w
                    if not finished_s:
                        dut.packetListS.value = chunk_s
                        dut.packetLimitS.value = length_s
                    if not finished_l1:
                        dut.packetListL1.value = chunk_l1
                        dut.packetLimitL1.value = length_l1
                    
                    
                    #print("IN VSLUE",dut.DataInE.value)
                    dut.counter_reset.value = dut.counter_reset.value+1
                    # await Timer(1, units='ps')
                    #await Edge(dut.ts_end)
                    await Edge(dut.simend)
                    start_index += max_dut_len
                    


    #dut.counter_reset.value = dut.counter_reset.value+1
    #print("FINISH",dut.finish.value)
    #await Edge(dut.simend)
    dut.finish.value = 1
    await Timer(1000, units='ps') 
    
    dut._log.info("TEST COMPLETED")
"""
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
"""