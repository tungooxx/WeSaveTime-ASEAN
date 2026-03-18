#!/usr/bin/env python3
"""Build the SUMO .net.xml from node/edge/tll XML files using netconvert."""

import os
import subprocess
import sys

SUMO_DIR = os.path.dirname(os.path.abspath(__file__))

def find_netconvert():
    """Locate the netconvert binary."""
    # 1. Check SUMO_HOME
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        nc = os.path.join(sumo_home, "bin", "netconvert")
        if os.path.isfile(nc) or os.path.isfile(nc + ".exe"):
            return nc

    # 2. Check eclipse-sumo pip package
    try:
        import sumolib
        sumo_bin = os.path.join(os.path.dirname(sumolib.__file__), "..", "bin")
        nc = os.path.join(os.path.abspath(sumo_bin), "netconvert")
        if os.path.isfile(nc) or os.path.isfile(nc + ".exe"):
            return nc
    except ImportError:
        pass

    # 3. Try PATH
    import shutil
    nc = shutil.which("netconvert")
    if nc:
        return nc

    print("ERROR: netconvert not found. Install SUMO or set SUMO_HOME.")
    sys.exit(1)


def build():
    netconvert = find_netconvert()
    print(f"Using netconvert: {netconvert}")

    cmd = [
        netconvert,
        "--node-files", os.path.join(SUMO_DIR, "hanoi_hk.nod.xml"),
        "--edge-files", os.path.join(SUMO_DIR, "hanoi_hk.edg.xml"),
        "--tllogic-files", os.path.join(SUMO_DIR, "hanoi_hk.tll.xml"),
        "--output-file", os.path.join(SUMO_DIR, "hanoi_hk.net.xml"),
        # Coordinates are geographic (lon, lat) — project to UTM
        "--proj", "+proj=utm +zone=48 +datum=WGS84 +units=m +no_defs",
        "--no-turnarounds", "true",
        "--junctions.join", "false",
        "--tls.default-type", "static",
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    net_file = os.path.join(SUMO_DIR, "hanoi_hk.net.xml")
    if os.path.isfile(net_file):
        size_kb = os.path.getsize(net_file) / 1024
        print(f"SUCCESS: {net_file} ({size_kb:.1f} KB)")
    else:
        print("FAILED: net file not generated")
        sys.exit(1)


if __name__ == "__main__":
    build()
