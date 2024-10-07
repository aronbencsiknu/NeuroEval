from python_files import generate_packets as gp
net, routing_matrices, routing_map, mapping, _, val_set, final_accuracy, _, _, _, _ = gp.snn_init(dut=None)
pac, exp = gp.delay_experiment(net, routing_matrices, routing_map, mapping, val_set, idx)