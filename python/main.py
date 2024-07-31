import hashlib
import torch

# -------------------------
# -------METHODS-----------
# -------------------------

def reset_spike_record_and_hooks():
  global spike_record, hooks

  # Clear the spike_record dictionary
  spike_record = {}

  # Remove existing hooks if they are already registered
  if 'hooks' in globals():
    for hook in hooks:
      hook.remove()
      hooks = []

# Function to create a hook that records spikes
def create_spike_hook(layer_name):
  def hook(module, input, output):
    if layer_name not in spike_record:
      spike_record[layer_name] = []
    spike_record[layer_name].append(output[0].detach().cpu())
  return hook


def reset_spike_record_and_hooks():
  global spike_record, hooks

  # Clear the spike_record dictionary
  spike_record = {}

  # Remove existing hooks if they are already registered
  if 'hooks' in globals():
    for hook in hooks:
      hook.remove()
      hooks = []

# Function to create a hook that records spikes
def create_spike_hook(layer_name):
  def hook(module, input, output):
    if layer_name not in spike_record:
      spike_record[layer_name] = []
    spike_record[layer_name].append(output[0].detach().cpu())
  return hook

def bundle_target_cores(source_core, target_cores, min_reps):
  res = []
  new_target_cores = []
  for idx, (target_core, reps) in enumerate(target_cores):
    res.append(target_core)
    if reps - min_reps > 0:
      new_target_cores.append((target_core, reps - min_reps))

  return res, new_target_cores

def remove_skipped_packets(source_core, target_cores, buffer_map):
  new_target_cores = []
  for idx, (target_core, reps) in enumerate(target_cores):
    if str(source_core)+str(target_core) in buffer_map:
      new_target_cores.append((target_core, reps - int(buffer_map[str(source_core)+str(target_core)])))
    else:
      new_target_cores.append((target_core, reps))

  return new_target_cores

def bundle_target_cores(source_core, target_cores, min_reps):
  res = []
  new_target_cores = []
  for idx, (target_core, reps) in enumerate(target_cores):
    res.append(target_core)
    if reps - min_reps > 0:
      new_target_cores.append((target_core, reps - min_reps))

  return res, new_target_cores

def remove_skipped_packets(source_core, target_cores, buffer_map):
  new_target_cores = []
  for idx, (target_core, reps) in enumerate(target_cores):
    if str(source_core)+str(target_core) in buffer_map:
      new_target_cores.append((target_core, reps - int(buffer_map[str(source_core)+str(target_core)])))
    else:
      new_target_cores.append((target_core, reps))

  return new_target_cores

# -------------------------
# -------METHODS-----------
# -------------------------

def main():
    # Dictionary to store spikes from each layer

    spike_record = {}
    hooks = []


    reset_spike_record_and_hooks()

    # Attach hooks automatically to all Leaky layers
    for name, module in net.named_modules():
        if isinstance(module, snn.Leaky) or isinstance(module, snn.RSynaptic):
            hooks.append(module.register_forward_hook(create_spike_hook(name)))

    routing_matrices = {}
    routing_map = {}

    for layer_name, size in mem_potential_sizes.items():
        routing_matrix = torch.zeros((num_steps, size))
        for idx in range(size):
            if layer_name not in routing_matrices:
                routing_id = layer_name +"-"+ str(idx)
                source_core = neuron_to_core[routing_id]

                downstream_nodes = list(G.successors(layer_name))

                target_cores = []
                for downstream_node in downstream_nodes:
                    if downstream_node != "output":
                        target_cores.extend(NIR_to_cores[downstream_node])

                # Remove skipped packets
                target_cores = remove_skipped_packets(source_core, target_cores, buffer_map)
                bundled_core_to_cores = []
                while len(target_cores) > 0:
                    _, minimum = target_cores[0]
                    for target_core, reps in target_cores:
                        if reps < minimum:
                            minimum = reps

                    bcc, target_cores = bundle_target_cores(source_core, target_cores, minimum)
                    bundled_core_to_cores.append((bcc, minimum))
                packet_information = []
                for bcc, reps in bundled_core_to_cores:
                    packet_information.append((source_core, bcc, reps))
                    h = int(hashlib.shake_256(routing_id.encode()).hexdigest(2), 16)
                    routing_map[h] = packet_information
                    for t in range(num_steps):
                        routing_matrix[t][idx] = h

    routing_matrices[layer_name] = routing_matrix

    #print(routing_matrices['lif1'])
    #print(routing_map)

    exp = torch.mul(routing_matrices['lif1'], spike_record['lif1'])

    temp = exp[1]
    non_zero_values = temp[temp != 0]


    packets = []
    for hashes in non_zero_values:
        packets.append(routing_map[int(hashes)])

    print(packets)


if __name__ == '__main__':
    main()