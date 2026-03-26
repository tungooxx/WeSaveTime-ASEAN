#!/usr/bin/env python3
"""
FlowMind AI - TLS Diagnostic Tool

Scans ALL TLS in the SUMO network and categorizes them:
  1. Which are real multi-road intersections with proper multi-phase programs (already good for AI)
  2. Which are real multi-road intersections with BAD single-phase programs (fixable → AI potential)
  3. Which are genuinely single-road (ped crossings, median breaks → leave on default)

This tells you exactly how many TLS the AI could control if phase programs were fixed.

Usage:
    python tls_diagnostic.py --net sumo/danang/danang.net.xml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import sumolib


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class TLSDiagnostic:
    id: str
    # Network topology
    num_incoming_edges: int = 0
    num_outgoing_edges: int = 0
    num_incoming_lanes: int = 0
    total_connections: int = 0
    incoming_edge_names: list[str] = field(default_factory=list)
    # Phase program
    num_phases: int = 0
    num_green_phases: int = 0
    phase_states: list[str] = field(default_factory=list)
    has_conflicting_phases: bool = False  # True = phases give green to different directions
    # Geometry
    width_m: float = 0.0
    junction_type: str = ""
    # Classification
    category: str = ""
    recommendation: str = ""
    ai_potential: str = ""
    # Roundabout
    near_roundabout: bool = False
    roundabout_dist: float = 999.0
    # Clustering
    nearby_tls_count: int = 0
    is_clustered: bool = False


def analyze_phase_conflict(phases: list) -> bool:
    """Check if different phases give green to DIFFERENT directions.
    
    If all phases have the same green pattern, there's no real conflict
    separation — it's effectively single-phase.
    
    Returns True if phases actually separate conflicting movements.
    """
    green_patterns = set()
    for phase in phases:
        state = phase.state if hasattr(phase, 'state') else str(phase)
        # Extract which positions get green
        pattern = tuple(i for i, c in enumerate(state) if c in ('G', 'g'))
        if pattern:  # skip all-red/all-yellow phases
            green_patterns.add(pattern)
    return len(green_patterns) >= 2


def count_distinct_green_phases(phases: list) -> int:
    """Count phases that are actual green phases (not yellow/allred transitions)."""
    count = 0
    for phase in phases:
        state = phase.state if hasattr(phase, 'state') else str(phase)
        has_green = any(c in ('G', 'g') for c in state)
        all_yr = all(c in ('y', 'Y', 'r', 'R') for c in state)
        if has_green and not all_yr:
            count += 1
    return count


def get_unique_road_names(net, node, edges) -> list[str]:
    """Get unique road names from edges, falling back to edge IDs."""
    names = []
    seen = set()
    for edge in edges:
        eid = edge.getID()
        if eid.startswith(":"):
            continue
        name = edge.getName() or eid
        # Clean up OSM-style names (remove #segment suffixes)
        base_name = name.split("#")[0].strip()
        if base_name and base_name not in seen:
            names.append(base_name)
            seen.add(base_name)
    return names


def get_unique_road_count(net, node) -> int:
    """Count unique roads (not edges) meeting at this junction.
    
    Multiple edges from the same road (e.g., road segments) should count as one.
    Uses road name to deduplicate.
    """
    road_names = set()
    for edge in list(node.getIncoming()) + list(node.getOutgoing()):
        eid = edge.getID()
        if eid.startswith(":"):
            continue
        name = edge.getName() or eid
        base = name.split("#")[0].strip()
        road_names.add(base)
    return len(road_names)


def run_diagnostic(net_file: str, roundabout_radius: float = 300.0,
                   cluster_radius: float = 30.0) -> list[TLSDiagnostic]:
    """Scan all TLS and produce diagnostic report."""
    
    net = sumolib.net.readNet(os.path.abspath(net_file), withPrograms=True)
    
    # ── Find roundabout centers ────────────────────────────────────
    ra_centers: list[tuple[float, float]] = []
    for ra in net.getRoundabouts():
        xs, ys = [], []
        for nid in ra.getNodes():
            try:
                n = net.getNode(nid)
                x, y = n.getCoord()
                xs.append(x); ys.append(y)
            except Exception:
                pass
        if xs:
            ra_centers.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    
    # ── Get all TLS coords for clustering ──────────────────────────
    tls_coords: dict[str, tuple[float, float]] = {}
    for tls in net.getTrafficLights():
        tid = tls.getID()
        try:
            node = net.getNode(tid)
            tls_coords[tid] = node.getCoord()
        except Exception:
            pass
    
    # ── Analyze each TLS ───────────────────────────────────────────
    results: list[TLSDiagnostic] = []
    
    for tls in net.getTrafficLights():
        tid = tls.getID()
        diag = TLSDiagnostic(id=tid)
        
        # ── Node topology ──────────────────────────────────────────
        try:
            node = net.getNode(tid)
            diag.junction_type = node.getType()
            
            incoming = [e for e in node.getIncoming() if not e.getID().startswith(":")]
            outgoing = [e for e in node.getOutgoing() if not e.getID().startswith(":")]
            
            diag.num_incoming_edges = len(incoming)
            diag.num_outgoing_edges = len(outgoing)
            diag.num_incoming_lanes = sum(e.getLaneNumber() for e in incoming)
            diag.incoming_edge_names = get_unique_road_names(net, node, incoming)
            
            # Junction width
            shape = node.getShape()
            if shape and len(shape) >= 2:
                xs = [p[0] for p in shape]
                ys = [p[1] for p in shape]
                diag.width_m = round(max(max(xs) - min(xs), max(ys) - min(ys)), 1)
            
        except Exception:
            pass
        
        # ── Phase program analysis ─────────────────────────────────
        programs = tls.getPrograms()
        if programs:
            prog = list(programs.values())[0]
            phases = prog.getPhases()
            diag.num_phases = len(phases)
            diag.num_green_phases = count_distinct_green_phases(phases)
            diag.phase_states = [p.state for p in phases]
            diag.has_conflicting_phases = analyze_phase_conflict(phases)
        
        # ── Connections ────────────────────────────────────────────
        connections = tls.getConnections()
        diag.total_connections = len(connections)
        
        # ── Roundabout proximity ───────────────────────────────────
        if tid in tls_coords:
            tx, ty = tls_coords[tid]
            for cx, cy in ra_centers:
                dist = math.sqrt((tx - cx) ** 2 + (ty - cy) ** 2)
                if dist < diag.roundabout_dist:
                    diag.roundabout_dist = round(dist, 1)
            diag.near_roundabout = diag.roundabout_dist < roundabout_radius
        
        # ── Clustering ─────────────────────────────────────────────
        if tid in tls_coords:
            tx, ty = tls_coords[tid]
            nearby = 0
            for oid, (ox, oy) in tls_coords.items():
                if oid != tid:
                    if math.sqrt((tx - ox) ** 2 + (ty - oy) ** 2) < cluster_radius:
                        nearby += 1
            diag.nearby_tls_count = nearby
            diag.is_clustered = nearby >= 2
        
        # ── Classification ─────────────────────────────────────────
        diag = classify_tls(diag)
        results.append(diag)
    
    return results


def classify_tls(d: TLSDiagnostic) -> TLSDiagnostic:
    """Classify a TLS and determine AI control potential."""
    
    ie = d.num_incoming_edges
    gp = d.num_green_phases
    has_conflict = d.has_conflicting_phases
    lanes = d.num_incoming_lanes
    
    # ── Category 1: Already multi-phase with conflict separation ───
    if gp >= 2 and has_conflict and ie >= 2:
        if d.near_roundabout:
            d.category = "MULTI_PHASE_ROUNDABOUT"
            d.recommendation = "AI control possible but needs coordinated roundabout metering"
            d.ai_potential = "MEDIUM"
        elif d.is_clustered:
            d.category = "MULTI_PHASE_CLUSTERED"
            d.recommendation = "AI control — but verify not a split junction (may need merge)"
            d.ai_potential = "HIGH"
        else:
            d.category = "MULTI_PHASE_GOOD"
            d.recommendation = "AI control — ready to optimize"
            d.ai_potential = "HIGH"
    
    # ── Category 2: Multiple roads but single/uniform phase (BAD PROGRAM) ──
    elif ie >= 3 and (gp <= 1 or not has_conflict):
        d.category = "BAD_PROGRAM_MULTI_ROAD"
        d.recommendation = (
            f"FIX PROGRAM: {ie} incoming roads but only {gp} green phase(s) "
            f"with no conflict separation. Needs {min(ie, 4)}-phase program, "
            f"then AI control"
        )
        d.ai_potential = "HIGH_AFTER_FIX"
    
    elif ie == 2 and (gp <= 1 or not has_conflict) and lanes >= 4:
        d.category = "BAD_PROGRAM_2ROAD_BUSY"
        d.recommendation = (
            f"FIX PROGRAM: 2-road intersection with {lanes} lanes but "
            f"only {gp} green phase(s). Needs 2-phase program (one per road), "
            f"then AI control"
        )
        d.ai_potential = "MEDIUM_AFTER_FIX"
    
    # ── Category 3: Single road, single phase — true ped crossing ──
    elif ie <= 1 and gp <= 1:
        d.category = "PED_CROSSING"
        d.recommendation = "Leave on SUMO default cycling (fixed timer ped crossing)"
        d.ai_potential = "NONE"
    
    # ── Category 4: 2 roads, few lanes, single phase — minor intersection ──
    elif ie == 2 and lanes <= 3 and gp <= 1:
        d.category = "MINOR_INTERSECTION"
        d.recommendation = (
            "Minor intersection — 2-phase program possible but low impact. "
            "Leave on SUMO default or low-priority fix"
        )
        d.ai_potential = "LOW"
    
    # ── Category 5: Roundabout entry ───────────────────────────────
    elif d.near_roundabout and gp <= 1:
        d.category = "ROUNDABOUT_ENTRY"
        d.recommendation = "Roundabout entry — yield-based flow is typically better"
        d.ai_potential = "LOW"
    
    # ── Category 6: Clustered fragment ─────────────────────────────
    elif d.is_clustered and gp <= 1:
        d.category = "CLUSTERED_FRAGMENT"
        d.recommendation = (
            f"Clustered with {d.nearby_tls_count} other TLS within 30m — "
            f"likely a split junction. Consider merging (increase junctions.join-dist)"
        )
        d.ai_potential = "NONE_UNTIL_MERGED"
    
    # ── Category 7: Multi-phase but uniform (same pattern all phases) ──
    elif gp >= 2 and not has_conflict:
        d.category = "UNIFORM_MULTI_PHASE"
        d.recommendation = (
            f"Has {gp} phases but all give green to same movements — "
            f"effectively single-phase. Needs program redesign"
        )
        d.ai_potential = "MEDIUM_AFTER_FIX"
    
    # ── Catch-all ──────────────────────────────────────────────────
    else:
        d.category = "OTHER"
        d.recommendation = f"Manual review needed: {ie} edges, {gp} green phases"
        d.ai_potential = "UNKNOWN"
    
    return d


def print_report(results: list[TLSDiagnostic]) -> None:
    """Print comprehensive diagnostic report."""
    
    print("=" * 100)
    print("  FlowMind AI - TLS Diagnostic Report")
    print("=" * 100)
    print(f"  Total TLS in network: {len(results)}")
    print()
    
    # ── Summary by category ────────────────────────────────────────
    categories = defaultdict(list)
    for d in results:
        categories[d.category].append(d)
    
    # Sort categories by AI potential
    potential_order = {
        "HIGH": 0, "HIGH_AFTER_FIX": 1, "MEDIUM_AFTER_FIX": 2,
        "MEDIUM": 3, "LOW": 4, "NONE": 5, "NONE_UNTIL_MERGED": 6,
        "UNKNOWN": 7,
    }
    
    print("  ┌─────────────────────────────────────────────────────────────────────┐")
    print("  │                        SUMMARY BY CATEGORY                         │")
    print("  ├──────────────────────────┬───────┬──────────────────────────────────┤")
    print(f"  │ {'Category':<24s} │ {'Count':>5s} │ {'AI Potential':<32s} │")
    print("  ├──────────────────────────┼───────┼──────────────────────────────────┤")
    
    for cat in sorted(categories.keys(), 
                      key=lambda c: potential_order.get(
                          categories[c][0].ai_potential, 99)):
        items = categories[cat]
        pot = items[0].ai_potential
        print(f"  │ {cat:<24s} │ {len(items):>5d} │ {pot:<32s} │")
    
    print("  └──────────────────────────┴───────┴──────────────────────────────────┘")
    print()
    
    # ── AI Control Summary ─────────────────────────────────────────
    ready_now = [d for d in results if d.ai_potential == "HIGH"]
    fixable_high = [d for d in results if d.ai_potential == "HIGH_AFTER_FIX"]
    fixable_med = [d for d in results if d.ai_potential in ("MEDIUM_AFTER_FIX", "MEDIUM")]
    no_potential = [d for d in results if d.ai_potential in ("NONE", "LOW", "NONE_UNTIL_MERGED")]
    
    print("  ┌─────────────────────────────────────────────────────────────────────┐")
    print("  │                       AI CONTROL POTENTIAL                          │")
    print("  ├─────────────────────────────────────────────────────────────────────┤")
    print(f"  │  Ready for AI NOW:           {len(ready_now):>3d} TLS                              │")
    print(f"  │  Fixable → HIGH potential:   {len(fixable_high):>3d} TLS (need multi-phase programs)  │")
    print(f"  │  Fixable → MEDIUM potential: {len(fixable_med):>3d} TLS (2-road or uniform phases)   │")
    print(f"  │  No AI potential:            {len(no_potential):>3d} TLS (ped crossings, fragments)   │")
    print(f"  │                                                                     │")
    print(f"  │  TOTAL AI-CONTROLLABLE:      {len(ready_now) + len(fixable_high) + len(fixable_med):>3d} TLS (after fixing programs)      │")
    print("  └─────────────────────────────────────────────────────────────────────┘")
    print()
    
    # ── Detail: Ready for AI now ───────────────────────────────────
    if ready_now:
        print("─" * 100)
        print(f"  READY FOR AI NOW ({len(ready_now)} TLS)")
        print("─" * 100)
        print(f"  {'TLS ID':<50s} {'GP':>3s} {'IE':>3s} {'Lanes':>5s} {'Width':>6s} {'Roads'}")
        print(f"  {'─'*50} {'─'*3} {'─'*3} {'─'*5} {'─'*6} {'─'*30}")
        for d in sorted(ready_now, key=lambda x: x.num_incoming_lanes, reverse=True):
            roads = " / ".join(d.incoming_edge_names[:3])
            print(f"  {d.id:<50s} {d.num_green_phases:>3d} {d.num_incoming_edges:>3d} "
                  f"{d.num_incoming_lanes:>5d} {d.width_m:>5.0f}m {roads[:30]}")
        print()
    
    # ── Detail: Fixable HIGH potential ─────────────────────────────
    if fixable_high:
        print("─" * 100)
        print(f"  FIX PROGRAM → HIGH AI POTENTIAL ({len(fixable_high)} TLS)")
        print("  These are REAL multi-road intersections with BAD single-phase programs")
        print("─" * 100)
        print(f"  {'TLS ID':<50s} {'GP':>3s} {'IE':>3s} {'Lanes':>5s} {'Width':>6s} {'Roads'}")
        print(f"  {'─'*50} {'─'*3} {'─'*3} {'─'*5} {'─'*6} {'─'*30}")
        for d in sorted(fixable_high, key=lambda x: x.num_incoming_lanes, reverse=True):
            roads = " / ".join(d.incoming_edge_names[:3])
            print(f"  {d.id:<50s} {d.num_green_phases:>3d} {d.num_incoming_edges:>3d} "
                  f"{d.num_incoming_lanes:>5d} {d.width_m:>5.0f}m {roads[:30]}")
        print()
        print("  To fix: regenerate phase programs for these TLS with proper conflict separation")
        print("  e.g., a 4-road intersection needs 2-4 phases, one per non-conflicting movement")
        print()
    
    # ── Detail: Fixable MEDIUM potential ───────────────────────────
    if fixable_med:
        print("─" * 100)
        print(f"  FIX PROGRAM → MEDIUM AI POTENTIAL ({len(fixable_med)} TLS)")
        print("─" * 100)
        print(f"  {'TLS ID':<50s} {'GP':>3s} {'IE':>3s} {'Lanes':>5s} {'Cat':<25s} {'Roads'}")
        print(f"  {'─'*50} {'─'*3} {'─'*3} {'─'*5} {'─'*25} {'─'*30}")
        for d in sorted(fixable_med, key=lambda x: x.num_incoming_lanes, reverse=True):
            roads = " / ".join(d.incoming_edge_names[:2])
            print(f"  {d.id:<50s} {d.num_green_phases:>3d} {d.num_incoming_edges:>3d} "
                  f"{d.num_incoming_lanes:>5d} {d.category:<25s} {roads[:30]}")
        print()
    
    # ── Detail: No potential ───────────────────────────────────────
    if no_potential:
        print("─" * 100)
        print(f"  NO AI POTENTIAL — LEAVE ON SUMO DEFAULT ({len(no_potential)} TLS)")
        print("─" * 100)
        # Group by sub-category
        sub_cats = defaultdict(list)
        for d in no_potential:
            sub_cats[d.category].append(d)
        for cat, items in sorted(sub_cats.items()):
            print(f"  {cat} ({len(items)}):")
            for d in items[:5]:  # Show first 5 per category
                roads = " / ".join(d.incoming_edge_names[:2]) or "(no named roads)"
                print(f"    {d.id:<50s} IE={d.num_incoming_edges} L={d.num_incoming_lanes} {roads[:25]}")
            if len(items) > 5:
                print(f"    ... and {len(items) - 5} more")
            print()
    
    # ── Phase program examples for fixable TLS ─────────────────────
    if fixable_high:
        print("─" * 100)
        print("  EXAMPLE: What's wrong with these phase programs?")
        print("─" * 100)
        for d in fixable_high[:3]:
            print(f"\n  TLS: {d.id}")
            print(f"  Roads: {', '.join(d.incoming_edge_names[:4])}")
            print(f"  Incoming edges: {d.num_incoming_edges}, Lanes: {d.num_incoming_lanes}")
            print(f"  Current phases ({d.num_phases} total, {d.num_green_phases} green):")
            for i, state in enumerate(d.phase_states):
                g_count = sum(1 for c in state if c in ('G', 'g'))
                r_count = sum(1 for c in state if c in ('r', 'R'))
                y_count = sum(1 for c in state if c in ('y', 'Y'))
                is_green = any(c in ('G', 'g') for c in state)
                marker = "GREEN" if is_green else "trans"
                print(f"    Phase {i}: [{marker:>5s}] G={g_count} R={r_count} Y={y_count}  {state}")
            
            if not d.has_conflicting_phases:
                print(f"  PROBLEM: All green phases give green to the SAME movements!")
                print(f"  FIX: Need separate phases for conflicting directions")
                print(f"       e.g., Phase A: NS green + EW red, Phase B: NS red + EW green")
        print()
    
    # ── Recommended action plan ────────────────────────────────────
    print("=" * 100)
    print("  RECOMMENDED ACTION PLAN")
    print("=" * 100)
    print(f"""
  Step 1: Keep current 10 multi-phase TLS under AI control (already working)
  
  Step 2: Fix phase programs for {len(fixable_high)} high-potential TLS
          These are real intersections that netconvert gave bad programs.
          Use netconvert --tls.guess or manually define phase programs.
          After fix: AI can control {len(ready_now) + len(fixable_high)} TLS total.
  
  Step 3: Optionally fix {len(fixable_med)} medium-potential TLS
          These are 2-road intersections or uniform-phase TLS.
          Lower priority — fix only if they cause congestion.
          After fix: AI can control {len(ready_now) + len(fixable_high) + len(fixable_med)} TLS total.
  
  Step 4: Leave {len(no_potential)} TLS on SUMO default
          Ped crossings, fragments, roundabout entries.
          No AI control needed — SUMO default cycling is correct.
""")
    print("=" * 100)


# ── CLI ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="FlowMind AI - TLS Diagnostic Tool")
    ap.add_argument("--net", required=True, help="SUMO .net.xml file")
    ap.add_argument("--roundabout-radius", type=float, default=300.0,
                    help="Radius to detect roundabout-adjacent TLS (default: 300m)")
    ap.add_argument("--cluster-radius", type=float, default=30.0,
                    help="Radius to detect clustered TLS (default: 30m)")
    args = ap.parse_args()
    
    if not os.path.isfile(args.net):
        print(f"ERROR: Network file not found: {args.net}")
        sys.exit(1)
    
    results = run_diagnostic(
        args.net,
        roundabout_radius=args.roundabout_radius,
        cluster_radius=args.cluster_radius,
    )
    print_report(results)


if __name__ == "__main__":
    main()
