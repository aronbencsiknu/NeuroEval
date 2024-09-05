import cocotb
from cocotb.triggers import Timer, Edge
import numpy as np
import sys
import snntorch as snn

from python_files.options import Variables
from python_files.options import Specs
from python_files import generate_packets as gp
from python_files import utils
from python_files.metrics import Metrics
from torch import stack
import snntorch.functional as SF
import torch

v = Variables()
s = Specs()


num_sent_messages = 0
num_recieved_messages = 0
expanded_packets = []
max_dut_len = 2000

wait_multiplier = 700

all_not_sent = 0
all_unexpected = 0

@cocotb.test()
async def testbench(dut):

    global expanded_packets
    global all_not_sent
    global all_unexpected
    
    fruits = None
    veggies = None
    mtrcs = Metrics()
    net, routing_matrices, routing_map, mapping, _, val_set, _, final_accuracy, metrics_old= gp.snn_init(dut=None)
    
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
        
        

        if ts_counter >= v.num_steps:
            #print("TARGET SMPL", sample_counter)
            _, target = val_set[sample_counter]
            targets.append(target)
            ts_counter = 0
            #sample_counter += 1
            asd = stack(output_whole, dim=0).tolist()
            all_outputs.append(asd)

            output_whole = []
            if sample_counter == 19:
                with torch.no_grad():
                    
                    x = torch.tensor(all_outputs)
                    x = x.permute(1, 0, 2)
                    print("shape before", x.shape)
                    x = x[-v.recall_duration:]
                    y = torch.tensor(targets)
                    print("all outputs", x.shape)
                    print("targets:",y.shape)
                    acc = SF.acc.accuracy_rate(x, y)

                    preds = mtrcs.return_predicted(x)
                    mtrcs.perf_measure(y, preds)
                    print("RESULTS", preds, targets)
                    print()
                    print("NEW ACCURACY", acc)
                    print("TP, TN, FP, FN")
                    print(metrics_old.get_scores())
                    print(mtrcs.get_scores())
                    
                    print()
                    print("MULTIPLIER", wait_multiplier)
                    print("ALL NOT DELIVERED", all_not_sent)
                    print("ALL UNEXPECTED", all_unexpected)
                    break
            
            sample_counter += 1

        

        if ts_counter == 0:
            x, _ = val_set[sample_counter]
            get_ts_data.init_membranes()
            
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

            temp = "timestep "+ str(ts_counter)+" : sample "+ str(sample_counter)
            #print(temp)
            # dut._log.info("--------------------------------------")
            dut._log.info(temp)
            # dut._log.info("--------------------------------------")
            
            main_index = 0
            within_ts_counter = 0
            all_in_ts = 0
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
                all_in_ts += amount_to_wait
                await Timer(amount_to_wait*wait_multiplier, units='ps') # allowed trasmit time

                if within_ts_counter == 0:
                    veggies = np.array(dut.veggies.value)
                
                within_ts_counter += 1
                main_index += max_dut_len

            fruits = np.array(dut.fruits.value)
            dut.packet_expand_done = 0
            dut.empty.value = 1
            await Edge(dut.empty)

            dut.packet_expand_done = 1
            print("All in ts", all_in_ts)
            print("Not sent",np.sum(fruits), fruits.shape)

            all_not_sent += np.sum(fruits)

            print("Unexpected",np.sum(veggies), veggies.shape)
            all_unexpected += np.sum(veggies)

        ts_counter+=1

    dut.finish.value = 1
    await Timer(1, units='ps')
    
    dut._log.info("TEST COMPLETED")
