import cocotb
from cocotb.triggers import Timer, Edge
import numpy as np

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

    def return_chunk(packets, idx):
         length = len(packets)

         if idx >=length - 1:
              return None, None, True

         if idx + max_dut_len > length:
              
              chunk = [int(element, base=2) for element in packets[idx:length]] + [0]*(max_dut_len-(length - idx))
              return chunk, length - idx, False
         
         chunk = [int(element, base=2) for element in packets[idx:idx + max_dut_len]]
         return chunk, max_dut_len, False

    for i in range(30,33):

        
        if not (len(packets[s.EAST][i]) == 0 and 
                len(packets[s.NORTH][i]) == 0 and 
                len(packets[s.WEST][i]) == 0 and 
                len(packets[s.SOUTH][i]) == 0 and
                len(packets[s.L1][i]) == 0):

            temp = "HELLO"+ str(i)
            dut._log.info("--------------------------------------")
            dut._log.info(temp)
            dut._log.info("--------------------------------------")
            
            main_index = 0

            index_e = 0
            index_n = 0
            index_w = 0
            index_s = 0
            index_l1 = 0

            while True:
                
                # no append to but just send lol
                _, length_e, finished_e = return_chunk(packets[s.EAST][i], main_index)
                _, length_n, finished_n = return_chunk(packets[s.NORTH][i], main_index)
                _, length_w, finished_w = return_chunk(packets[s.WEST][i], main_index)
                _, length_s, finished_s = return_chunk(packets[s.SOUTH][i], main_index)
                _, length_l1, finished_l1 = return_chunk(packets[s.L1][i], main_index)

                chunk_e, _, sent_all_e = return_chunk(packets[s.EAST][i], index_e)
                chunk_n, _, sent_all_n = return_chunk(packets[s.NORTH][i], index_n)
                chunk_w, _, sent_all_w = return_chunk(packets[s.WEST][i], index_w)
                chunk_s, _, sent_all_s = return_chunk(packets[s.SOUTH][i], index_s)
                chunk_l1, _, sent_all_l1 = return_chunk(packets[s.L1][i], index_l1)
                
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

                dut.counter_reset.value = dut.counter_reset.value+1
                await Edge(dut.simend)
                await Timer(1, units='ps')

                # await Timer(70000, units='ps') # allowed trasmit time

                if not finished_e:
                    num_sent_e = dut.packetCounterE.value
                    index_e += max_dut_len - (max_dut_len - num_sent_e)

                if not finished_n:
                    num_sent_n = dut.packetCounterN.value
                    index_n += max_dut_len - (max_dut_len - num_sent_n)

                if not finished_w:
                    num_sent_w = dut.packetCounterW.value
                    index_w += max_dut_len - (max_dut_len - num_sent_w)

                if not finished_s:
                    num_sent_s = dut.packetCounterS.value
                    index_s += max_dut_len - (max_dut_len - num_sent_s)

                if not finished_l1:
                    num_sent_l1 = dut.packetCounterL1.value
                    index_l1 += max_dut_len - (max_dut_len - num_sent_l1)

                main_index += max_dut_len

                print()
                print("Main Index:", main_index)
                print("Index E:", index_e)
                print("Index N:", index_n)
                print("Index W:", index_w)
                print("Index S:", index_s)
                print("Index L1:", index_l1)

                current_time = cocotb.utils.get_sim_time()
                cocotb.log.info(f"Current simulation time: {current_time} ns")

    #dut.packet_expand_done.value = 0
    #dut.sync.value = 1
    #await Edge(dut.fruits)
    #data_not_sent_e = packets[s.EAST][i][num_sent_e:]
    print("Sent All E:", sent_all_e, )
    print("Sent All N:", sent_all_n)
    print("Sent All W:", sent_all_w)
    print("Sent All S:", sent_all_s)
    print("Sent All L1:", sent_all_l1)
    print()
    print("Length of Packets EAST:", len(packets[s.EAST][i]))
    print("Length of Packets NORTH:", len(packets[s.NORTH][i]))
    print("Length of Packets WEST:", len(packets[s.WEST][i]))
    print("Length of Packets SOUTH:", len(packets[s.SOUTH][i]))
    print("Length of Packets L1:", len(packets[s.L1][i]))
    print()
    #print("REMAINING",dut.fruits.value)
    test = np.array(dut.fruits.value)
    print(np.sum(test))

    #await Edge(dut.simend)

    dut.finish.value = 1
    await Timer(1, units='ps')
    
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