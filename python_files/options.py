import torch
import argparse

class Variables(object):
    def __init__(self):
        self.num_inputs = 2
        self.num_hidden1 = 100
        self.num_outputs = 2
        self.num_steps = 100
        self.core_capacity = 50
        self.num_epochs = 30
        self.lr = 1e-4
        self.target_fr = 0.5
        self.bs = 16
        self.num_cores = 5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Specs(object):
    def __init__(self):
        self.ADDR_W = 5
        self.MSG_W = 10
        self.EAST, self.NORTH, self.WEST, self.SOUTH, self.L1 = range(5)
        self.NUM_PACKETS_P_INJ = 20

        self.SID = 0b00001
        self.E_MASK = 0b10000
        self.N_MASK = 0b01000
        self.W_MASK = 0b00100
        self.S_MASK = 0b00010