#!/usr/bin/env python3
"""
ATCS optimizer: compute green splits for each junction based on demand.
"""

import json, math, time
from pathlib import Path

def load_map(json_path="config/junctions.json"):
    with open(json_path,'r') as f:
        return json.load(f)['junctions']

def compute_demands(stats_json, approach_weights=None):
    """
    Convert per-frame counts into per-junction demand values.
    For demo, just use total vehicles counted in stats_json summary.
    """
    with open(stats_json,'r') as f:
        stats=json.load(f)
    total = stats['summary']['total_vehicles_detected']
    return total

def allocate_greens(demands, base_min=7, cycle_time=90):
    """
    demands: dict of {junction_id: demand_value}
    returns: dict {junction_id: green_time}
    """
    # ensure nonzero
    sum_demand = sum(max(1,d) for d in demands.values())
    res={}
    for j, d in demands.items():
        share = (d/sum_demand) if sum_demand>0 else 1/len(demands)
        green = base_min + share*(cycle_time - base_min*len(demands))
        res[j] = round(green,1)
    return res

def compute_offsets(junctions, avg_speed=10.0):
    """
    Compute phase offsets (green wave) based on neighbor distance/avg_speed.
    avg_speed in m/s (10 m/s = ~36 km/h)
    """
    offsets={}
    for j in junctions:
        for nb in j['neighbors']:
            delay = nb['distance']/avg_speed
            offsets[f"{j['id']}->{nb['to']}"] = round(delay,1)
    return offsets

if __name__=="__main__":
    junctions=load_map()
    # Example: load demands from sample stats files
    demands={"J1":50,"J2":30,"J3":80,"J4":20,"J5":10,"J6":60}
    greens=allocate_greens(demands)
    offsets=compute_offsets(junctions)
    print("Green splits (sec):",greens)
    print("Offsets (sec):",offsets)