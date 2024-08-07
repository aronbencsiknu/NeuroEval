import numpy as np
import numpy.random as rd
import torch
from torch.utils.data import Dataset, DataLoader
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt

class BinaryNavigationDataset(Dataset):
    def __init__(self, batch_size, seq_len, n_neuron, recall_duration, p_group, f0=0.5,
                 n_cues=7, t_cue=100, t_interval=150, n_input_symbols=4, length=100):
        
        self.data = []
        self.targets = []
        for i in range(length):
            data, targets = self.generate_click_task_data(
                batch_size, seq_len, n_neuron, recall_duration, p_group, f0,
                n_cues, t_cue, t_interval, n_input_symbols
            )
            self.data.append(data)
            self.targets.append(targets)

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
    def generate_poisson_noise_np(self, prob_pattern, freezing_seed=None):
        if isinstance(prob_pattern, list):
            return [self.generate_poisson_noise_np(pb, freezing_seed=freezing_seed) for pb in prob_pattern]

        shp = prob_pattern.shape

        if freezing_seed is not None:
            rng = rd.RandomState(freezing_seed)
        else:
            rng = rd.RandomState()

        spikes = prob_pattern > rng.rand(prob_pattern.size).reshape(shp)
        return spikes

    def generate_click_task_data(self, batch_size, seq_len, n_neuron, recall_duration, p_group, f0=0.5,
                                n_cues=7, t_cue=100, t_interval=150,
                                n_input_symbols=4):
        t_seq = seq_len
        n_channel = n_neuron // n_input_symbols

        prob_choices = np.array([p_group, 1 - p_group], dtype=np.float32)
        idx = rd.choice([0, 1])
        probs = np.zeros((2), dtype=np.float32)
        probs[0] = prob_choices[idx]
        probs[1] = prob_choices[1 - idx]

        cue_assignments = np.zeros((n_cues), dtype=np.int32)
        cue_assignments = rd.choice([0, 1], n_cues, p=probs)

        input_spike_prob = np.zeros((t_seq, n_neuron))
        d_silence = t_interval - t_cue
        # for b in range(batch_size):
        #     for k in range(n_cues):
        #         c = cue_assignments[b, k]
        #         idx = k
        #         input_spike_prob[b, d_silence + idx * t_interval:d_silence + idx * t_interval + t_cue, c * n_channel:(c + 1) * n_channel] = f0

        for k in range(n_cues):
            c = cue_assignments[k]
            idx = k
            input_spike_prob[d_silence + idx * t_interval:d_silence + idx * t_interval + t_cue, c * n_channel:(c + 1) * n_channel] = f0

        input_spike_prob[-recall_duration:, 2 * n_channel:3 * n_channel] = f0
        input_spike_prob[:, 3 * n_channel:] = f0 / 4
        data = self.generate_poisson_noise_np(input_spike_prob)
        
        # I don't really get this yet but it works
        target_nums = np.zeros((seq_len), dtype=np.int32)
        target_nums[:] = np.transpose(np.tile(np.sum(cue_assignments, axis=0) > int(n_cues / 2), (seq_len, 1)))
        targets = target_nums[0]

        return data, targets

# # Parameters
# n_in = 20
# t_cue_spacing = 20
# silence_duration = 100
# recall_duration = 20
# seq_len = int(t_cue_spacing * 7 + silence_duration + recall_duration)
# batch_size = 10
# input_f0 = 40. / 100.
# p_group = 0.3
# n_cues = 7
# t_cue = 20
# t_interval = t_cue_spacing
# n_input_symbols = 4

# # Create dataset and dataloader
# dataset = BinaryNavigationDataset(batch_size, seq_len, n_in, recall_duration, p_group, input_f0, n_cues, t_cue, t_interval, n_input_symbols)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# print(len(dataset))

# # Visualize the data
# for spk_data, target_data in dataloader:
#     data = spk_data[0]
#     print(spk_data.size())

#     fig = plt.figure(facecolor="w", figsize=(10, 5))
#     ax = fig.add_subplot(111)

#     splt.raster(data, ax, s=1.5, c="black")

#     plt.title("Input Layer")
#     plt.xlabel("Time step")
#     plt.ylabel("Neuron Number")
#     plt.show()
#     #break  # Only display the first batch
