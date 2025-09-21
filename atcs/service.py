#!/usr/bin/env python3
"""
ATCS service: compute demands from mapped stats files and provide optimizer endpoint.
This replaces the hardcoded demands with real data from processed videos.
"""

import json, os, time
from pathlib import Path
from optimizer import allocate_greens, load_map

# Paths (adjust relative to atcs/ directory)
MAPPINGS_FILE = Path("../backend/mappings.json")
JUNCTIONS_FILE = Path("../config/junctions.json")

# Fallback to absolute paths if relative paths don't work
if not MAPPINGS_FILE.exists():
    MAPPINGS_FILE = Path("backend/mappings.json")
if not JUNCTIONS_FILE.exists():
    JUNCTIONS_FILE = Path("config/junctions.json")

def compute_demand_from_stats(stats_path, uid=None, fps=25, window_seconds=60):
    """
    Prefer unique counts from backend/aggregates/<uid>_unique_counts.json if exists.
    Otherwise fall back to stats file flow_windows or summary.
    """
    # Try to use unique counts first if uid is provided
    if uid:
        agg_path = Path("../backend/aggregates") / f"{uid}_unique_counts.json"
        if not agg_path.exists():
            agg_path = Path("backend/aggregates") / f"{uid}_unique_counts.json"
        
        if agg_path.exists():
            try:
                # read last N snapshots and average unique_count_60s
                with open(agg_path, 'r') as f:
                    lines = f.read().strip().splitlines()
                    if not lines:
                        print(f"Empty aggregates file: {agg_path}")
                    else:
                        last = [json.loads(L) for L in lines[-3:]]  # average last 3
                        vals = [l.get('unique_count_60s', 0) for l in last]
                        if vals:
                            demand = float(sum(vals) / len(vals))
                            print(f"Using unique counts for demand: {demand:.1f}")
                            return demand
            except Exception as e:
                print(f"Error reading aggregates file {agg_path}: {e}")
    
    # Fallback to previous implementation using flow_windows
    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        flows = stats.get('flow_windows', [])
        if not flows:
            # Fallback to total vehicle count if no flow windows
            total_vehicles = stats.get('summary', {}).get('total_vehicles_detected', 0)
            return float(total_vehicles / 10)  # rough conversion to demand
        
        # Take average of last 3 windows (or all if fewer than 3)
        recent_flows = flows[-3:] if len(flows) >= 3 else flows
        vals = [f['vehicles_per_minute'] for f in recent_flows]
        
        return float(sum(vals) / len(vals)) if vals else 0.0
        
    except Exception as e:
        print(f"Error reading stats from {stats_path}: {e}")
        return 0.0

def get_junction_demands():
    """
    Read mappings and compute demand for each junction from mapped stats files.
    """
    # Load junction IDs
    try:
        junctions = load_map(str(JUNCTIONS_FILE))
        junction_ids = [j['id'] for j in junctions]
    except Exception as e:
        print(f"Error loading junctions: {e}")
        junction_ids = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
    
    # Initialize demands
    demands = {jid: 0.0 for jid in junction_ids}
    
    # Load mappings
    if not MAPPINGS_FILE.exists():
        print("No mappings file found, using default demands")
        return {jid: 10.0 for jid in junction_ids}
    
    try:
        with open(MAPPINGS_FILE, 'r') as f:
            data = json.load(f)
        mappings = data.get('mappings', [])
    except Exception as e:
        print(f"Error reading mappings: {e}")
        return {jid: 10.0 for jid in junction_ids}
    
    # Compute demand per junction from mapped stats
    for mapping in mappings:
        stats_path = mapping.get('stats', '')
        junction = mapping.get('junction', '')
        fps = mapping.get('fps', 25)
        
        if not os.path.exists(stats_path):
            print(f"Stats file not found: {stats_path}")
            continue
            
        if junction not in demands:
            print(f"Unknown junction: {junction}")
            continue
        
        try:
            # Extract uid from mapping for aggregate lookup
            uid = mapping.get('uid', None)  # Get stored uid
            if not uid:
                # Fallback: extract uid from stats filename
                stats_filename = Path(stats_path).name
                if stats_filename.endswith('_stats.json'):
                    uid = stats_filename[:-11]
            
            demand = compute_demand_from_stats(stats_path, uid=uid, fps=fps)
            demands[junction] += demand
            print(f"Junction {junction}: +{demand:.1f} demand from {stats_path} (uid: {uid})")
        except Exception as e:
            print(f"Error computing demand for {junction} from {stats_path}: {e}")
    
    # If all demands are zero, use small defaults to avoid division by zero
    if sum(demands.values()) == 0:
        print("All demands are zero, using default values")
        demands = {jid: 10.0 for jid in junction_ids}
    
    return demands

def get_optimized_plan(live_demand_overrides=None):
    """
    Get current junction demands and return optimized green time allocation.
    Now includes demand values in the response.
    live_demand_overrides: dict of {junction: demand} to override mapped demands
    """
    demands = get_junction_demands()
    
    # Apply live demand overrides
    if live_demand_overrides:
        for junction, live_demand in live_demand_overrides.items():
            if junction in demands:
                demands[junction] = live_demand
                print(f"Live override: {junction} = {live_demand:.1f}")
    
    print(f"Current demands: {demands}")
    
    greens = allocate_greens(demands)
    print(f"Optimized greens: {greens}")
    
    # Return both demands and greens
    return {
        "greens": greens,
        "demands": demands,
        "timestamp": time.time(),
        "total_demand": sum(demands.values()),
        "live_overrides": live_demand_overrides or {}
    }

# Standalone execution for testing
if __name__ == "__main__":
    print("Testing ATCS service...")
    plan = get_optimized_plan()
    print(f"Signal plan: {plan}")