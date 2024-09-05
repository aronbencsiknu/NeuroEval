# A Benchmarking Framework for Neuromorphic Network-on-Chips
## About
Master's thesis project
## Instructions
1. Paste the Verilog files of the TaBuLa NoC (not provided) in the root directory.
2. Install [Intel Questa](https://www.intel.com/content/www/us/en/software/programmable/quartus-prime/questa-edition.html).
3. Build the conda environment: `conda env create -f environment.yml`.

Packet delivery delays can be measured by running Python  `python test_runner_delay.py`. 
Packet delivery inconsistencies caused by high-frequency sensors can be investigated by running Python  `python test_runner_accelerated.py`. 

This will first train an SNN model using snnTorch, or use a pre-trained model and then use the recorded spike traces to drive the NoC simulation. It uses Cocotb to drive the Questa simulation of the NoC model. Cocotb is configured to launch from a Python runner instead of a Makefile.
