# This file is public domain, it can be freely copied without restrictions.
# SPDX-License-Identifier: CC0-1.0

# test_runner.py

import os
from pathlib import Path

from cocotb.runner import get_runner


def test_my_design_runner():
    sim = os.getenv("SIM", "questa")

    proj_path = Path(__file__).resolve().parent

    sources = list(proj_path.glob("*.v"))

    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel="switch_5x5_XY",
    )

    runner.test(hdl_toplevel="switch_5x5_XY", test_module="testbench,", waves=True)


if __name__ == "__main__":
    test_my_design_runner()
