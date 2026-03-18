#!/usr/bin/env python3
"""
FlowMind AI - Build Hai Chau District SUMO network from OpenStreetMap.

Uses a pre-downloaded map.osm for Hai Chau District (Da Nang downtown core).

Steps:
  1. Use local map.osm (Hai Chau District)
  2. Convert to SUMO network with netconvert
  3. Generate polygon overlays (buildings, parks, water) with polyconvert
  4. Generate traffic demand with randomTrips.py + duarouter
  5. Create .sumocfg ready for sumo-gui
"""

import os
import subprocess
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════
# Hai Chau District — Da Nang's downtown core
#   Pre-downloaded OSM: map.osm (bounds 16.047-16.065, 108.205-108.224)
#   Bach Dang riverside, Han Market, Da Nang Cathedral
#   Major roads: Tran Phu, Le Duan, Nguyen Van Linh, Hai Phong, Hung Vuong
# ══════════════════════════════════════════════════════════════════════

# Output files — input is the pre-downloaded map.osm
OSM_FILE    = os.path.join(SCRIPT_DIR, "map.osm")
NET_FILE    = os.path.join(SCRIPT_DIR, "danang.net.xml")
POLY_FILE   = os.path.join(SCRIPT_DIR, "danang.poly.xml")
TRIPS_FILE  = os.path.join(SCRIPT_DIR, "danang.trips.xml")
ROUTES_FILE = os.path.join(SCRIPT_DIR, "danang.rou.xml")
VTYPES_FILE = os.path.join(SCRIPT_DIR, "danang.vtypes.xml")
CFG_FILE    = os.path.join(SCRIPT_DIR, "danang.sumocfg")

# Typemap for polyconvert
TYPEMAP_FILE = os.path.join(SCRIPT_DIR, "typemap.xml")


def find_tool(name):
    """Find a SUMO tool (binary or Python script)."""
    # Check PATH
    found = shutil.which(name)
    if found:
        return found
    # Check pip sumo package (two possible locations)
    try:
        import sumolib
        base = os.path.dirname(os.path.dirname(sumolib.__file__))
        for sub in ["tools", os.path.join("sumo", "tools")]:
            candidate = os.path.join(base, sub, name)
            if os.path.isfile(candidate):
                return candidate
    except ImportError:
        pass
    # Check SUMO_HOME
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home:
        for subdir in ["bin", "tools"]:
            c = os.path.join(sumo_home, subdir, name)
            if os.path.isfile(c) or os.path.isfile(c + ".exe"):
                return c
    return name  # hope it's on PATH


def run(cmd, desc=""):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {' '.join(cmd[:5])}...")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        # Print last few lines only
        lines = result.stdout.strip().split("\n")
        for line in lines[-10:]:
            print(f"  {line}")
    if result.stderr:
        lines = result.stderr.strip().split("\n")
        for line in lines[-15:]:
            print(f"  {line}")
    if result.returncode != 0:
        print(f"  WARNING: exit code {result.returncode}")
    return result.returncode == 0


# ══════════════════════════════════════════════════════════════════════
# Step 1: Verify local OSM file exists
# ══════════════════════════════════════════════════════════════════════

def check_osm():
    if os.path.isfile(OSM_FILE):
        size_mb = os.path.getsize(OSM_FILE) / (1024 * 1024)
        print(f"  Using local OSM: {OSM_FILE} ({size_mb:.1f} MB)")
        return True
    print(f"  ERROR: {OSM_FILE} not found!")
    print("  Download Hai Chau District from https://www.openstreetmap.org/export")
    return False


# ══════════════════════════════════════════════════════════════════════
# Step 2: Convert OSM to SUMO network
# ══════════════════════════════════════════════════════════════════════

def build_network():
    netconvert = find_tool("netconvert")
    print(f"Using netconvert: {netconvert}")

    cmd = [
        netconvert,
        "--osm-files", OSM_FILE,
        "--output-file", NET_FILE,
        # Road type defaults
        "--osm.speedlimit-none", "13.89",  # 50 km/h default
        # Traffic lights
        "--tls.default-type", "static",
        "--tls.guess", "true",
        "--tls.guess.threshold", "100",
        # Junction handling
        "--junctions.join", "true",
        "--junctions.join-dist", "20",
        "--junctions.corner-detail", "5",
        # Lane and road geometry
        "--geometry.remove", "true",
        "--roundabouts.guess", "true",
        "--ramps.guess", "true",
        # Remove isolated edges (dead-ends shorter than 50m)
        "--remove-edges.isolated", "true",
        "--keep-edges.by-vclass", "passenger,motorcycle,bus,truck",
        # No turnarounds at minor junctions
        "--no-turnarounds.except-deadend", "true",
        # Preserve street names from OSM
        "--output.street-names", "true",
        # Projection
        "--proj.utm", "true",
        # Clean output
        "--no-warnings", "true",
    ]

    return run(cmd, "Building SUMO network from OSM")


# ══════════════════════════════════════════════════════════════════════
# Step 3: Generate polygon overlays (buildings, water, parks)
# ══════════════════════════════════════════════════════════════════════

def build_polygons():
    polyconvert = find_tool("polyconvert")

    # Create typemap for polygon styling
    typemap = """<?xml version="1.0" encoding="UTF-8"?>
<polygonTypes>
    <polygonType id="building"    name="building"    color="0.6,0.5,0.4,0.6" layer="-2"/>
    <polygonType id="landuse.residential" name="residential" color="0.8,0.8,0.7,0.3" layer="-4"/>
    <polygonType id="landuse.commercial"  name="commercial"  color="0.9,0.8,0.7,0.3" layer="-4"/>
    <polygonType id="landuse.industrial"  name="industrial"  color="0.7,0.7,0.7,0.3" layer="-4"/>
    <polygonType id="natural.water"       name="water"       color="0.4,0.6,0.8,0.7" layer="-3"/>
    <polygonType id="waterway"            name="waterway"    color="0.4,0.6,0.8,0.5" layer="-3"/>
    <polygonType id="leisure.park"        name="park"        color="0.4,0.7,0.3,0.5" layer="-3"/>
</polygonTypes>
"""
    with open(TYPEMAP_FILE, "w") as f:
        f.write(typemap)

    cmd = [
        polyconvert,
        "--osm-files", OSM_FILE,
        "--net-file", NET_FILE,
        "--type-file", TYPEMAP_FILE,
        "--output-file", POLY_FILE,
        "--no-warnings", "true",
    ]

    return run(cmd, "Building polygon overlays")


# ══════════════════════════════════════════════════════════════════════
# Step 4: Create Vietnamese vehicle types
# ══════════════════════════════════════════════════════════════════════

def create_vehicle_types():
    vtypes = """<?xml version="1.0" encoding="UTF-8"?>
<!-- FlowMind AI - Vietnamese vehicle types for Da Nang -->
<additional>
    <!-- Motorbikes (dominant in Vietnamese traffic) -->
    <vType id="motorbike" vClass="motorcycle" length="2.2" width="0.8"
           maxSpeed="11.11" accel="3.0" decel="4.5" sigma="0.7"
           speedFactor="1.0" speedDev="0.15"
           color="0.9,0.7,0.2" guiShape="motorcycle"/>

    <vType id="motorbike2" vClass="motorcycle" length="2.0" width="0.7"
           maxSpeed="12.50" accel="3.5" decel="5.0" sigma="0.8"
           speedFactor="1.1" speedDev="0.2"
           color="0.2,0.6,0.9" guiShape="motorcycle"/>

    <!-- Cars -->
    <vType id="car" vClass="passenger" length="4.5" width="1.8"
           maxSpeed="13.89" accel="2.5" decel="4.0" sigma="0.5"
           speedFactor="1.0" speedDev="0.1"
           color="0.2,0.6,0.9" guiShape="passenger"/>

    <vType id="car_suv" vClass="passenger" length="4.8" width="2.0"
           maxSpeed="13.89" accel="2.2" decel="3.8" sigma="0.5"
           speedFactor="1.0" speedDev="0.1"
           color="0.8,0.2,0.2" guiShape="passenger/sedan"/>

    <vType id="taxi" vClass="passenger" length="4.5" width="1.8"
           maxSpeed="13.89" accel="2.5" decel="4.0" sigma="0.4"
           speedFactor="0.9" speedDev="0.1"
           color="1.0,1.0,0.2" guiShape="passenger/sedan"/>

    <!-- Buses -->
    <vType id="bus" vClass="bus" length="12.0" width="2.5"
           maxSpeed="11.11" accel="1.5" decel="3.0" sigma="0.4"
           speedFactor="0.8" speedDev="0.05"
           color="0.2,0.8,0.3" guiShape="bus"/>

    <!-- Trucks -->
    <vType id="truck" vClass="truck" length="10.0" width="2.5"
           maxSpeed="8.33" accel="1.2" decel="3.0" sigma="0.4"
           speedFactor="0.7" speedDev="0.1"
           color="0.7,0.3,0.3" guiShape="truck"/>

    <vType id="delivery" vClass="truck" length="6.0" width="2.2"
           maxSpeed="11.11" accel="1.8" decel="3.5" sigma="0.5"
           speedFactor="0.85" speedDev="0.1"
           color="0.6,0.4,0.2" guiShape="truck"/>
</additional>
"""
    with open(VTYPES_FILE, "w") as f:
        f.write(vtypes)
    print(f"  Vehicle types -> {VTYPES_FILE}")
    return True


# ══════════════════════════════════════════════════════════════════════
# Step 4b: Generate edge weights (busy vs quiet roads)
# ══════════════════════════════════════════════════════════════════════

WEIGHTS_PREFIX = os.path.join(SCRIPT_DIR, "danang.weights")

def create_edge_weights():
    """Create edge weight files so major roads (more lanes) attract more traffic.

    Vietnamese traffic reality:
      - 5+ lane arterials: packed with motorbikes & cars
      - 3-4 lane roads: busy secondary corridors
      - 2 lane roads: moderate neighborhood traffic
      - 1 lane alleys: occasional access only
    """
    import sumolib
    net = sumolib.net.readNet(NET_FILE)

    src_file = WEIGHTS_PREFIX + ".src.xml"
    dst_file = WEIGHTS_PREFIX + ".dst.xml"

    lines = ['<meandata>\n', '    <interval begin="0" end="3600">\n']
    for edge in net.getEdges():
        eid = edge.getID()
        if eid.startswith(":"):
            continue
        lanes = edge.getLaneNumber()
        speed = edge.getSpeed()  # m/s

        # Weight by lanes and speed — arterials dominate
        if lanes >= 5:
            w = 20.0
        elif lanes >= 4:
            w = 12.0
        elif lanes >= 3:
            w = 6.0
        elif lanes >= 2:
            w = 2.0
        else:
            w = 0.5

        # Boost high-speed roads (highways/expressways)
        if speed > 16.0:  # > 60 km/h
            w *= 2.0

        lines.append(f'        <edge id="{eid}" value="{w:.1f}"/>\n')

    lines.append('    </interval>\n')
    lines.append('</meandata>\n')

    content = "".join(lines)
    for f_path in (src_file, dst_file):
        with open(f_path, "w") as f:
            f.write(content)

    print(f"  Edge weights -> {src_file}")
    print(f"                  {dst_file}")

    # Print distribution
    counts = {20: 0, 12: 0, 6: 0, 2: 0, 0.5: 0}
    for edge in net.getEdges():
        if edge.getID().startswith(":"):
            continue
        lanes = edge.getLaneNumber()
        if lanes >= 5: counts[20] += 1
        elif lanes >= 4: counts[12] += 1
        elif lanes >= 3: counts[6] += 1
        elif lanes >= 2: counts[2] += 1
        else: counts[0.5] += 1
    print(f"  Road hierarchy:")
    print(f"    Arterials (5+ lanes, w=20): {counts[20]} edges")
    print(f"    Major (4 lanes, w=12):      {counts[12]} edges")
    print(f"    Secondary (3 lanes, w=6):   {counts[6]} edges")
    print(f"    Local (2 lanes, w=2):        {counts[2]} edges")
    print(f"    Alleys (1 lane, w=0.5):      {counts[0.5]} edges")
    return True


# ══════════════════════════════════════════════════════════════════════
# Step 5: Generate traffic demand
# ══════════════════════════════════════════════════════════════════════

def generate_demand():
    randomTrips = find_tool("randomTrips.py")
    duarouter = find_tool("duarouter")

    # Hai Chau District: dense downtown, 10,000 vehicles for 1-hour peak
    # ~68% motorbike, ~20% car, ~6% bus, ~6% truck
    trip_files = []

    vehicle_configs = [
        # (prefix, vtype, count, fringe_factor)
        ("ma_", "motorbike",   4500, 1),   # main motorbike wave
        ("mx_", "motorbike2",  2300, 2),   # secondary motorbike wave
        ("ca_", "car",         1200, 5),   # sedans
        ("sv_", "car_suv",      500, 5),   # SUVs
        ("tx_", "taxi",         500, 5),   # taxis (Grab/Mai Linh)
        ("bs_", "bus",          400, 10),  # city buses
        ("tk_", "truck",        300, 9),   # heavy trucks
        ("dv_", "delivery",     300, 5),   # delivery vans
    ]

    for prefix, vtype, count, fringe in vehicle_configs:
        trip_file = os.path.join(SCRIPT_DIR, f"danang.{prefix}.trips.xml")
        trip_files.append(trip_file)

        cmd = [
            sys.executable, randomTrips,
            "-n", NET_FILE,
            "-o", trip_file,
            "--additional-file", VTYPES_FILE,
            "--trip-attributes", f'type="{vtype}"',
            "--prefix", prefix,
            "-b", "0",
            "-e", "3600",
            "-p", str(round(3600 / count, 2)),
            "--fringe-factor", str(fringe),
            "--validate",
            "--remove-loops",
            "--random",
            "--weights-prefix", WEIGHTS_PREFIX,  # major roads get way more traffic
        ]
        run(cmd, f"Generating {count} {vtype} trips")

    # Merge all trip files into one
    print("\nMerging trip files...")
    merge_trips(trip_files, TRIPS_FILE)

    # Route with duarouter
    cmd = [
        duarouter,
        "-n", NET_FILE,
        "--route-files", TRIPS_FILE,
        "--additional-files", VTYPES_FILE,
        "-o", ROUTES_FILE,
        "--ignore-errors", "true",
        "--no-warnings", "true",
        "--repair", "true",
    ]
    return run(cmd, "Computing routes with duarouter")


def merge_trips(trip_files, output):
    """Merge multiple trip XML files into one, sorted by depart time."""
    import xml.etree.ElementTree as ET

    trips = []
    for f in trip_files:
        if not os.path.isfile(f):
            continue
        tree = ET.parse(f)
        root = tree.getroot()
        for trip in root:
            if trip.tag in ("trip", "vehicle"):
                depart = float(trip.get("depart", "0"))
                trips.append((depart, ET.tostring(trip, encoding="unicode")))

    trips.sort(key=lambda x: x[0])

    with open(output, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<trips>\n')
        for _, xml_str in trips:
            f.write(f"    {xml_str}\n")
        f.write('</trips>\n')

    print(f"  Merged {len(trips)} trips -> {output}")


# ══════════════════════════════════════════════════════════════════════
# Step 5b: Add candidate TLS at busy uncontrolled junctions
# ══════════════════════════════════════════════════════════════════════

CANDIDATE_FILE = os.path.join(SCRIPT_DIR, "candidate_tls.json")
MAX_CANDIDATES = 15
SIM_SAMPLE_STEPS = 2000  # ~1000 sim-seconds at step_length=0.5

def add_candidate_tls():
    """Run a baseline simulation to find the most CONGESTED uncontrolled
    junctions, then add TLS at those locations for the RL agent to evaluate.

    Instead of picking candidates by road size (hardcoded), we let the
    actual traffic simulation tell us where congestion is worst.

    Flow:
      1. Discover all uncontrolled junctions with 3+ incoming roads
      2. Run SUMO baseline for ~1000s, measure wait time & queue at each
      3. Score junctions by real congestion data
      4. Add TLS at top-N most congested via netconvert
      5. RL agent then decides: keep the new TLS active or turn it OFF
    """
    import json
    import traci
    import sumolib

    sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", ".."))
    from src.simulation.tls_metadata import (
        discover_uncontrolled_junctions,
        analyze_junctions_with_traci,
    )

    # ── 1. Existing TLS ────────────────────────────────────────────
    net = sumolib.net.readNet(NET_FILE, withPrograms=True)
    existing_tls = {tls.getID() for tls in net.getTrafficLights()}
    print(f"  Existing TLS: {len(existing_tls)}")

    # ── 2. Discover all candidate junctions ────────────────────────
    all_junctions = discover_uncontrolled_junctions(
        NET_FILE, min_incoming_edges=3)
    print(f"  Uncontrolled junctions (3+ roads): {len(all_junctions)}")

    if not all_junctions:
        print("  No candidate junctions found.")
        with open(CANDIDATE_FILE, "w") as f:
            json.dump({"existing_tls": list(existing_tls), "candidates": []}, f)
        return True

    # ── 3. Run baseline simulation to measure congestion ───────────
    print(f"  Running baseline simulation ({SIM_SAMPLE_STEPS} steps) "
          f"to measure congestion...")

    binary = find_tool("sumo")
    cmd = [binary, "-c", CFG_FILE,
           "--seed", "42",
           "--no-step-log", "true",
           "--no-warnings", "true",
           "--time-to-teleport", "300"]

    # Clean stale connections
    for label in ["candidate_analysis"]:
        try:
            traci.getConnection(label).close()
        except Exception:
            pass

    traci.start(cmd, label="candidate_analysis")
    conn = traci.getConnection("candidate_analysis")

    # Warm up for 200 steps (~100s) so vehicles enter the network
    for _ in range(200):
        conn.simulationStep()

    # Measure congestion at all uncontrolled junctions
    results = analyze_junctions_with_traci(
        conn, all_junctions, sample_steps=SIM_SAMPLE_STEPS)
    conn.close()

    print(f"  Congestion analysis complete: {len(results)} junctions scored")

    # ── 4. Pick top-N most congested ───────────────────────────────
    # Results are already sorted by congestion_score (worst first)
    top = results[:MAX_CANDIDATES]

    print(f"\n  Top {len(top)} congested junctions (candidates for TLS):")
    print(f"  {'Junction':<50s} {'Score':>6s} {'Wait':>7s} {'Queue':>6s} {'Roads':>5s}")
    print(f"  {'-'*50} {'-'*6} {'-'*7} {'-'*6} {'-'*5}")
    for r in top:
        jid = r["junction_id"][:48]
        print(f"  {jid:<50s} {r['congestion_score']:>5.3f} "
              f"{r['avg_wait']:>6.1f}s {r['avg_queue']:>5.1f} "
              f"{r['incoming_edges']:>5d}")

    # Filter: only add junctions with meaningful congestion
    top = [r for r in top if r["congestion_score"] > 0.1]
    if not top:
        print("\n  No significantly congested junctions found.")
        with open(CANDIDATE_FILE, "w") as f:
            json.dump({"existing_tls": list(existing_tls), "candidates": []}, f)
        return True

    # ── 5. Add TLS via netconvert ──────────────────────────────────
    node_ids = [r["junction_id"] for r in top]

    netconvert = find_tool("netconvert")
    cmd = [
        netconvert,
        "--sumo-net-file", NET_FILE,
        "--output-file", NET_FILE,
        "--tls.set", ",".join(node_ids),
        "--tls.default-type", "static",
        "--no-warnings", "true",
    ]
    success = run(cmd, "Adding candidate TLS at congested junctions")
    if not success:
        print("  WARNING: netconvert failed to add candidate TLS")
        with open(CANDIDATE_FILE, "w") as f:
            json.dump({"existing_tls": list(existing_tls), "candidates": []}, f)
        return True

    # ── 6. Save metadata ──────────────────────────────────────────
    candidate_data = {
        "existing_tls": list(existing_tls),
        "candidates": [
            {
                "id": r["junction_id"],
                "x": r["x"], "y": r["y"],
                "incoming_edges": r["incoming_edges"],
                "incoming_lanes": r["incoming_lanes"],
                "junction_type": r["junction_type"],
                "congestion_score": r["congestion_score"],
                "avg_wait": r["avg_wait"],
                "avg_queue": r["avg_queue"],
                "recommendation": r["recommendation"],
            }
            for r in top
        ],
    }
    with open(CANDIDATE_FILE, "w") as f:
        json.dump(candidate_data, f, indent=2)
    print(f"  Candidate metadata -> {CANDIDATE_FILE}")

    # Verify
    net2 = sumolib.net.readNet(NET_FILE, withPrograms=True)
    new_count = len(net2.getTrafficLights())
    print(f"  Total TLS after: {new_count} "
          f"(+{new_count - len(existing_tls)} candidates)")

    return True


# ══════════════════════════════════════════════════════════════════════
# Step 6: Create SUMO config
# ══════════════════════════════════════════════════════════════════════

def create_config():
    cfg = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="danang.net.xml"/>
        <route-files value="danang.rou.xml"/>
        <additional-files value="danang.poly.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="0.5"/>
    </time>
    <processing>
        <lateral-resolution value="0.8"/>
        <ignore-route-errors value="true"/>
    </processing>
    <gui_only>
        <gui-settings-file value="danang.view.xml"/>
    </gui_only>
    <report>
        <no-warnings value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>
"""
    with open(CFG_FILE, "w") as f:
        f.write(cfg)
    print(f"  Config -> {CFG_FILE}")

    # GUI view settings
    view = """<?xml version="1.0" encoding="UTF-8"?>
<viewsettings>
    <scheme name="FlowMind">
        <background backgroundColor="0.15,0.15,0.20"/>
    </scheme>
    <delay value="50"/>
</viewsettings>
"""
    view_file = os.path.join(SCRIPT_DIR, "danang.view.xml")
    with open(view_file, "w") as f:
        f.write(view)

    return True


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  FlowMind AI - Hai Chau District SUMO Network Builder")
    print("=" * 60)
    print(f"  Source: {OSM_FILE}")
    print()

    steps = [
        ("1. Check OSM file",            check_osm),
        ("2. Build SUMO network",        build_network),
        ("3. Build polygon overlays",    build_polygons),
        ("4a. Create vehicle types",     create_vehicle_types),
        ("4b. Create edge weights",      create_edge_weights),
        ("5. Generate traffic demand",   generate_demand),
        ("5b. Add candidate TLS",       add_candidate_tls),
        ("6. Create SUMO config",        create_config),
    ]

    for desc, func in steps:
        print(f"\n>>> {desc}")
        success = func()
        if not success:
            print(f"\n  FAILED at: {desc}")
            print("  Fix the issue and re-run this script.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("  BUILD COMPLETE!")
    print("=" * 60)
    net_size = os.path.getsize(NET_FILE) / (1024 * 1024)
    print(f"  Network: {net_size:.1f} MB")
    print(f"  Config:  {CFG_FILE}")
    print()
    print("  To run:")
    print(f'    sumo-gui "{CFG_FILE}"')
    print()


if __name__ == "__main__":
    main()
