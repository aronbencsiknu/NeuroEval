from python_files import generate_packets as gp
import json
from python_files.options import Variables

v = Variables()

net, routing_matrices, routing_map, mapping, _, val_set, _, final_accuracy, metrics_old, upstream_connectivity = gp.snn_init(dut=None)
data = gp.delay_experiment(net, routing_matrices, routing_map, mapping, val_set, 0)

def convert_to_json(data):
    # Create a dictionary with each timestep as a key
    json_data = {f"TS-{i+1}": timestep for i, timestep in enumerate(data)}
    
    # Convert the dictionary to JSON format with specific formatting
    return json.dumps(json_data, indent=4)

for i in range(v.num_cores):
    json_result = convert_to_json(data[i])
    with open('data_core-%d.json' % i, 'w') as f:
        f.write(json_result)

data_dict = dict(upstream_connectivity)

# Save to a JSON file
with open('upstream_conn.json', 'w') as json_file:
    json.dump(data_dict, json_file, indent=4)