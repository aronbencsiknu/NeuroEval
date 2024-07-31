import torch
import argparse

class Options(object):
    def __init__(self):
        # self.parser = argparse.ArgumentParser()
        # self.initialized = False
        # print("HI")

        self.num_inputs = 2
        self.num_hidden1 = 100
        self.num_outputs = 2
        self.num_steps = 100
        self.core_capacity = 50
        self.num_epochs = 150
        self.lr = 1e-4
        self.target_fr = 0.5
        self.bs = 16
        self.num_cores = 5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # def add_args(self):

    #     self.parser.add_argument("--num_inputs", type=int, default=2, help="Number of inputs")
    #     self.parser.add_argument("--num_hidden1", type=int, default=100, help="Number of hidden layer 1 neurons")
    #     self.parser.add_argument("--num_outputs", type=int, default=2, help="Number of outputs")
    #     self.parser.add_argument("--num_steps", type=int, default=100, help="Number of simulation timesteps")
    #     self.parser.add_argument("--core_capacity", type=int, default=50, help="Core capacity")

    #     # ----------------------------

    #     self.initialized = True

    # def parse(self):
    #     if not self.initialized:
    #         self.add_args()

    #     self.opt = self.parser.parse_args()

    #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    #     self.opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
    #     return self.opt