import cocotb
from cocotb.triggers import Timer, Edge
import numpy as np
import sys
import snntorch as snn

from python_files.options import Variables
from python_files.options import Specs
from python_files import generate_packets as gp
from python_files import utils
from torch import stack
import snntorch.functional as SF
import torch

v = Variables()
s = Specs()

num_sent_messages = 0
num_recieved_messages = 0
expanded_packets = []
max_dut_len = 1000

@cocotb.test()
async def testbench(dut):

    global expanded_packets
    fruits = None
    veggies = None
    #packets, expanded_packets = gp.snn_main(dut=None)
    net, routing_matrices, routing_map, mapping, _, val_set, _, final_accuracy = gp.snn_init(dut=None)
    # val_set = []
    # for data, target in val_set_full:
    #     if target == 1:
    #         val_set.append((data, target))
    net.eval()
    #sys.exit()
    get_ts_data = gp.DynamicInference(net)
    get_ts_data.attach_hooks()
    
    await Edge(dut.init_done)

    def return_chunk(packets, idx):
         length = len(packets)

         if idx >=length - 1:
              return None, 0, True

         if idx + max_dut_len > length:
              chunk = [int(element, base=2) for element in packets[idx:length]] + [0]*(max_dut_len-(length - idx))
              return chunk, length - idx, False
         
         chunk = [int(element, base=2) for element in packets[idx:idx + max_dut_len]]
         return chunk, max_dut_len, False
    
    sample_counter = 0
    ts_counter = 0
    
    output_whole = []
    all_outputs = []
    targets = []

    while True:
        
        if sample_counter == 20:
            with torch.no_grad():
                
                x = torch.tensor(all_outputs)
                x = x.permute(1, 0, 2)
                y = torch.tensor(targets)
                print("all outputs", x.shape)
                print("targets:",y.shape)
                acc = SF.acc.accuracy_rate(x[-v.recall_duration:], y)
                
                print("final_accuracy", final_accuracy)
                print("ACCURACY", acc)
                break
        if ts_counter >= v.num_steps:
            #print("TARGET SMPL", sample_counter)
            _, target = val_set[sample_counter]
            targets.append(target)
            ts_counter = 0
            sample_counter += 1
            asd = stack(output_whole, dim=0).tolist()
            all_outputs.append(asd)

            output_whole = []

        if ts_counter == 0:
            x, _ = val_set[sample_counter]
            get_ts_data.init_membranes()
            
            #print("DATA SMPL", sample_counter)
            
        #spike_record = {}
        #output_spikes, spk1, syn1, mem1, mem2 = net.forward_one_ts(x[i].to(v.device), spk1, syn1, mem1, mem2, time_first=True)
        #print("NUM STEPS",v.num_steps, "COUNTER", ts_counter)
        with torch.no_grad():
            spike_record, output = get_ts_data.advance_inference(x[ts_counter], skipped_spikes=fruits, add_spikes=veggies)
            

        output_whole.append(output)

        packets_in_ts = []
        for layer_name, _ in mapping.mem_potential_sizes.items():

                p = utils.dot_product(routing_matrices[layer_name], 
                                            spike_record[layer_name][0], # it is going to be 1 element long in this case (dynamic packet generation)
                                            routing_map,
                                            )
                
                packets_in_ts.extend(p)

        final_packets_dict = {
            s.EAST: [],
            s.NORTH: [],
            s.WEST: [],
            s.SOUTH: [],
            s.L1: []
        }
        temp, expanded_packets = utils.repeat_and_convert_packets(packets_in_ts, final_packets_dict, s.ADDR_W)

        for key in final_packets_dict:
            if key in temp:
                final_packets_dict[key] = temp[key] 

        packets = final_packets_dict

        #print(packets[s.EAST])
        
        if not (len(packets[s.EAST]) == 0 and 
                len(packets[s.NORTH]) == 0 and 
                len(packets[s.WEST]) == 0 and 
                len(packets[s.SOUTH]) == 0 and
                len(packets[s.L1]) == 0):

            temp = "HELLO"+ str(ts_counter)+":"+ str(sample_counter)
            dut._log.info("--------------------------------------")
            dut._log.info(temp)
            dut._log.info("--------------------------------------")
            
            main_index = 0
            within_ts_counter = 0
            while True:
                
                chunk_e, length_e, finished_e = return_chunk(packets[s.EAST], main_index)
                chunk_n, length_n, finished_n = return_chunk(packets[s.NORTH], main_index)
                chunk_w, length_w, finished_w = return_chunk(packets[s.WEST], main_index)
                chunk_s, length_s, finished_s = return_chunk(packets[s.SOUTH], main_index)
                chunk_l1, length_l1, finished_l1 = return_chunk(packets[s.L1], main_index)
                
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
                amount_to_wait = length_e + length_n + length_w + length_s
                #print("AMOUNT TO WAIT",amount_to_wait)
                await Timer(amount_to_wait*900, units='ps') # allowed trasmit time

                # record new spikes at the beginning of the timesteps
                #print("TIME BEFORE",cocotb.utils.get_sim_time())
                #await Edge(dut.finished_sending)
                print("TIME AFTER",cocotb.utils.get_sim_time())
                #await Timer(1, units='ps')
                if within_ts_counter == 0:
                    veggies = np.array(dut.veggies.value)
                
                
                within_ts_counter += 1
                main_index += max_dut_len
            #await Timer(100, units='ps')
            fruits = np.array(dut.fruits.value)
            dut.packet_expand_done = 0
            dut.empty.value = 1
            await Edge(dut.empty)

            dut.packet_expand_done = 1
            
            print("fruits",np.sum(fruits), fruits.shape)

            #veggies = np.array(dut.veggies.value)
            print("veggies",np.sum(veggies), veggies.shape)

            print()
            current_time = cocotb.utils.get_sim_time()
            cocotb.log.info(f"Current simulation time: {current_time} ns")
            #sys.exit()
        ts_counter+=1

    # #await Timer(100000, units='ps') # allowed trasmit time
    # print()
    # #print("REMAINING",dut.fruits.value)
    # fruits = np.array(dut.fruits.value)
    # print("fruits",np.sum(fruits))

    # veggies = np.array(dut.veggies.value)
    # print("veggies",np.sum(veggies))

    # #await Edge(dut.simend)

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