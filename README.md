# A Benchmarking Framework for Neuromorphic Network-on-Chips
## About
Master's thesis project
## Instructions
Requires the Verilog files of the TaBuLa NoC in the root directory!

Uses Cocotb to drive the Questa simulation of the NoC model. Cocotb is configured to launch from a python runner instead of a Makefile.

Start the simulation by starting the Python runner as `python test_runner.py`. This will first train an SNN model using snnTorch, then use the recorded spike-traces to drive the NoC simulation.
