import random
import os

def generate_dynamic_traffic(filepath, max_steps=3600):
    """
    Overwrites the routing file with randomized probabilistic flows.
    Simulates different traffic conditions (balanced, NS-rush, EW-rush).
    """
    with open(filepath, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes>\n')
        # Add realistic vehicle physics (imperfection in acceleration/sigma)
        f.write('    <vType id="standard_car" vClass="passenger" accel="2.6" decel="4.5" sigma="0.5" length="4.5" maxSpeed="15"/>\n')
        
        # Base probability (e.g., 0.05 = ~1 car every 20 seconds per route)
        p_base = random.uniform(0.02, 0.08) 
        
        # Randomly choose a traffic scenario for this specific episode
        scenarios = ["balanced", "ns_rush", "ew_rush"]
        scenario = random.choice(scenarios)
        
        # Every possible route in your crossroad environment
        routes = [
            ("in_N", "out_S"), ("in_N", "out_E"), ("in_N", "out_W"),
            ("in_S", "out_N"), ("in_S", "out_E"), ("in_S", "out_W"),
            ("in_W", "out_E"), ("in_W", "out_N"), ("in_W", "out_S"),
            ("int_E", "out_W"), ("int_E", "out_N"), ("int_E", "out_S")
        ]
        
        for i, (in_edge, out_edge) in enumerate(routes):
            prob = p_base
            
            # Apply realistic volume biases based on the scenario
            if scenario == "ns_rush" and in_edge in ["in_N", "in_S"]:
                prob += random.uniform(0.05, 0.15)
            elif scenario == "ew_rush" and in_edge in ["in_W", "int_E"]:
                prob += random.uniform(0.05, 0.15)
            
            # Inject noise so routes aren't perfectly identical
            prob = max(0.01, min(prob * random.uniform(0.8, 1.2), 0.3))
            
            f.write(f'    <flow id="flow_{i}" begin="0" end="{max_steps}" '
                    f'probability="{prob:.3f}" from="{in_edge}" to="{out_edge}" '
                    f'type="standard_car" departLane="best" departSpeed="max"/>\n')
            
        f.write('</routes>\n')