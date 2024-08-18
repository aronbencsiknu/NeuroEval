import cocotb
from cocotb.triggers import Timer, Edge
import numpy as np
import sys
import snntorch as snn

from python_files.options import Variables
from python_files.options import Specs
from python_files import generate_packets as gp
from python_files import utils

v = Variables()
s = Specs()

num_sent_messages = 0
num_recieved_messages = 0
expanded_packets = []
max_dut_len = 100

@cocotb.test()
async def testbench(dut):

    global expanded_packets

    #packets, expanded_packets = gp.snn_main(dut=None)
    net, routing_matrices, routing_map, mapping, train_set, val_set = gp.snn_init(dut=None)
    # spk1, syn1, mem1 = net.lif1.init_rsynaptic()
    # mem2 = net.lif2.init_leaky()
    # print(routing_matrices)

    # # Dictionary to store spikes from each layer

    # spike_record = {}
    # hooks = []

    # def reset_spike_record_and_hooks():
    #     global spike_record, hooks

    #     # Clear the spike_record dictionary
    #     spike_record = {}

    #     # Remove existing hooks if they are already registered
    #     if 'hooks' in globals():
    #         for hook in hooks:
    #             hook.remove()
    #             hooks = []

    # # Function to create a hook that records spikes
    # def create_spike_hook(layer_name):
    #     def hook(module, input, output):
    #         if layer_name not in spike_record:
    #             spike_record[layer_name] = []
    #         spike_record[layer_name].append(output[0].detach().cpu())
    #     return hook

    # reset_spike_record_and_hooks()

    # # Attach hooks automatically to all Leaky layers
    # for name, module in net.named_modules():
    #     if isinstance(module, snn.Leaky) or isinstance(module, snn.RSynaptic):
    #         hooks.append(module.register_forward_hook(create_spike_hook(name)))

    get_ts_data = gp.DynamicInference(net)
    get_ts_data.attach_hooks()
    
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
    
    max_iters = 20
    print("NUM STEPS",v.num_steps)
    sys.exit()
    for i in range(0,max_iters):

        x, _ = dataset[0]
        #spike_record = {}
        #output_spikes, spk1, syn1, mem1, mem2 = net.forward_one_ts(x[i].to(v.device), spk1, syn1, mem1, mem2, time_first=True)

        spike_record, _ = get_ts_data.advance_inference(x[i])

        print(spike_record)
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
                
                chunk_e, length_e, finished_e = return_chunk(packets[s.EAST], main_index)
                chunk_n, length_n, finished_n = return_chunk(packets[s.NORTH], main_index)
                chunk_w, length_w, finished_w = return_chunk(packets[s.WEST], main_index)
                chunk_s, length_s, finished_s = return_chunk(packets[s.SOUTH], main_index)
                chunk_l1, length_l1, finished_l1 = return_chunk(packets[s.L1], main_index)

                # chunk_e, sent_all_length_e, sent_all_e = return_chunk(packets[s.EAST][i], index_e)
                # chunk_n, sent_all_length_n, sent_all_n = return_chunk(packets[s.NORTH][i], index_n)
                # chunk_w, sent_all_length_w, sent_all_w = return_chunk(packets[s.WEST][i], index_w)
                # chunk_s, sent_all_length_s, sent_all_s = return_chunk(packets[s.SOUTH][i], index_s)
                # chunk_l1, sent_all_length_l1, sent_all_l1 = return_chunk(packets[s.L1][i], index_l1)

                # if i == max_iters-1:
                #     finished_e = sent_all_e
                #     finished_n = sent_all_n
                #     finished_w = sent_all_w
                #     finished_s = sent_all_s
                #     finished_l1 = sent_all_l1

                #     length_e = sent_all_length_e
                #     length_n = sent_all_length_n
                #     length_w = sent_all_length_w
                #     length_s = sent_all_length_s
                #     length_l1 = sent_all_length_l1

                
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
                # if i == max_iters-1:
                #     await Edge(dut.simend)
                #     await Timer(1, units='ps')
                # else:
                #     await Timer(30000, units='ps') # allowed trasmit time

                await Edge(dut.ts_end)
                await Timer(1, units='ps')

                #await Timer(60000, units='ps') # allowed trasmit time

                # if not finished_e:
                #     num_sent_e = dut.packetCounterE.value
                #     #index_e += max_dut_len - (max_dut_len - num_sent_e)
                #     index_e += num_sent_e
                #     total_to_send3 += num_sent_e
                #     not_delivered += length_e - num_sent_e
                # if not finished_n:
                #     num_sent_n = dut.packetCounterN.value
                #     #index_n += max_dut_len - (max_dut_len - num_sent_n)
                #     index_n += num_sent_n
                #     total_to_send3 += num_sent_n
                #     not_delivered += length_n - num_sent_n

                # if not finished_w:
                #     num_sent_w = dut.packetCounterW.value
                #     #index_w += max_dut_len - (max_dut_len - num_sent_w)
                #     index_w += num_sent_w
                #     total_to_send3 += num_sent_w
                #     not_delivered += length_w - num_sent_w

                # if not finished_s:
                #     num_sent_s = dut.packetCounterS.value
                #     #index_s += max_dut_len - (max_dut_len - num_sent_s)
                #     index_s += num_sent_s
                #     total_to_send3 += num_sent_s
                #     not_delivered += length_s - num_sent_s

                # if not finished_l1:
                #     num_sent_l1 = dut.packetCounterL1.value
                #     #index_l1 += max_dut_len - (max_dut_len - num_sent_l1)
                #     index_l1 += num_sent_l1
                #     total_to_send3 += num_sent_l1
                #     not_delivered += length_l1 - num_sent_l1

                main_index += max_dut_len
            #await Timer(100, units='ps') # allowed trasmit time
            dut.packet_expand_done = 0
            dut.empty.value = 1
            await Edge(dut.empty) # allowed trasmit time
            dut.packet_expand_done = 1
            fruits = np.array(dut.fruits.value)
            print("fruits",np.sum(fruits), fruits.shape)

            veggies = np.array(dut.veggies.value)
            print("veggies",np.sum(veggies), veggies.shape)
            #await Edge(dut.expandedPacketsList)
            # remaining_e = packets[s.EAST][i][index_e:]
            # remaining_n = packets[s.NORTH][i][index_n:]
            # remaining_w = packets[s.WEST][i][index_w:]
            # remaining_s = packets[s.SOUTH][i][index_s:]
            # remaining_l1 = packets[s.L1][i][index_l1:]

            #await Timer(100000, units='ps') # allowed trasmit time
            print()

            # if i != max_iters-1:
            #     print("before",len(packets[s.EAST][i+1]))
            #     packets[s.EAST][i+1] = remaining_e +  packets[s.EAST][i+1]
            #     packets[s.NORTH][i+1] = remaining_n + packets[s.NORTH][i+1]
            #     packets[s.WEST][i+1] = remaining_w + packets[s.WEST][i+1]
            #     packets[s.SOUTH][i+1] = remaining_s + packets[s.SOUTH][i+1]
            #     packets[s.L1][i+1] = remaining_l1 + packets[s.L1][i+1]
            #     print("after",len(packets[s.EAST][i+1]))

            # print("Main Index:", main_index)
            # print("not delivered: ", not_delivered)
            # print("Index E:", index_e, f"Length of packets[s.EAST][{i}] = {len(packets[s.EAST][i])}", f"Original size: {original_size_e}", f"Remaining: {len(remaining_e)}")
            # print("Index N:", index_n, f"Length of packets[s.NORTH][{i}] = {len(packets[s.NORTH][i])}", f"Original size: {original_size_n}", f"Remaining: {len(remaining_n)}")
            # print("Index W:", index_w, f"Length of packets[s.WEST][{i}] = {len(packets[s.WEST][i])}", f"Original size: {original_size_w}", f"Remaining: {len(remaining_w)}")
            # print("Index S:", index_s, f"Length of packets[s.SOUTH][{i}] = {len(packets[s.SOUTH][i])}", f"Original size: {original_size_s}", f"Remaining: {len(remaining_s)}")
            # print("Index L1:", index_l1, f"Length of packets[s.L1][{i}] = {len(packets[s.L1][i])}", f"Original size: {original_size_l1}", f"Remaining: {len(remaining_l1)}")



            current_time = cocotb.utils.get_sim_time()
            cocotb.log.info(f"Current simulation time: {current_time} ns")

    await Timer(100000, units='ps') # allowed trasmit time
    # remaining_e = packets[s.EAST][max_iters-1][index_e:]
    # print("Remaining E:", f"Length of remaining E = {len(remaining_e)}")

    # remaining_n = packets[s.NORTH][max_iters-1][index_n:]
    # print("Remaining N:", f"Length of remaining N = {len(remaining_n)}")

    # remaining_w = packets[s.WEST][max_iters-1][index_w:]
    # print("Remaining W:", f"Length of remaining W = {len(remaining_w)}")

    # remaining_s = packets[s.SOUTH][max_iters-1][index_s:]
    # print("Remaining S:", f"Length of remaining S = {len(remaining_s)}")

    # remaining_l1 = packets[s.L1][max_iters-1][index_l1:]
    # print("Remaining L1:", f"Length of remaining L1 = {len(remaining_l1)}")
    #dut.packet_expand_done.value = 0
    #dut.sync.value = 1
    #await Edge(dut.fruits)
    #data_not_sent_e = packets[s.EAST][i][num_sent_e:]
    # print("total to send:", total_to_send)
    # print("total to send2:", total_to_send2)
    # print("total to send3:", total_to_send3)
    # print("Sent All E:", sent_all_e, )
    # print("Sent All N:", sent_all_n)
    # print("Sent All W:", sent_all_w)
    # print("Sent All S:", sent_all_s)
    # print("Sent All L1:", sent_all_l1)
    #print()
    # print("Length of Packets EAST:", len(packets[s.EAST][i]))
    # print("Length of Packets NORTH:", len(packets[s.NORTH][i]))
    # print("Length of Packets WEST:", len(packets[s.WEST][i]))
    # print("Length of Packets SOUTH:", len(packets[s.SOUTH][i]))
    # print("Length of Packets L1:", len(packets[s.L1][i]))
    print()
    #print("REMAINING",dut.fruits.value)
    fruits = np.array(dut.fruits.value)
    print("fruits",np.sum(fruits))

    veggies = np.array(dut.veggies.value)
    print("veggies",np.sum(veggies))

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