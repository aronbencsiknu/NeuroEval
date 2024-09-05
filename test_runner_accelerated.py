# This file is public domain, it can be freely copied without restrictions.
# SPDX-License-Identifier: CC0-1.0

# test_runner.py

import os
from pathlib import Path

from cocotb.runner import get_runner


def tb_runner():

    hdl_toplevel_lang = os.getenv("VHDL_GPI_INTERFACE", "vhpi")
    sim = os.getenv("SIM", "questa")

    proj_path = Path(__file__).resolve().parent

    sources = list(proj_path.glob("*.v"))
    sources.extend(list(proj_path.glob("*.sv")))

    sim_args = ["-suppress", "vsim-3839"]


    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel="tb_mt_stage",
        
          # Pass the simulator arguments here,
    )

    runner.test(hdl_toplevel="tb_mt_stage", test_module="testbench_accelerated,", waves=True, test_args=["-suppress", "vsim-3839", "-suppress", "vsim-12003"])


if __name__ == "__main__":
    tb_runner()
