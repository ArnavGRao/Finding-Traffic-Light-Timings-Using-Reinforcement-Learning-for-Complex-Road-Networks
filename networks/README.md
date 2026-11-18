# Traffic demand and RESCO network files

## Single intersection synthetic demands

All demand files use the existing four-approach network and define a 3,600 s
episode. Pass one route file to the SUMO environment for each run:

- `single_intersection/demand_balanced_symmetric.rou.xml`: each approach
  receives 600 vehicles/hour (400 straight, 100 right, 100 left).
- `single_intersection/demand_asymmetric_peak_arterial.rou.xml`: north/south
  each receive 900 vehicles/hour; east/west each receive 250 vehicles/hour.
- `single_intersection/demand_stepwise_tidal.rou.xml`: balanced demand for
  0–1,200 s, north/south peak for 1,200–2,400 s, and east/west peak for
  2,400–3,600 s.

SUMO's seeded flow generation varies the individual vehicle departure times;
pass the evaluation seed to SUMO on every run. Use the same seed list for each
controller/model so comparisons are paired.

## RESCO 4×4 grid

`resco_4x4/` contains the upstream SUMO-RL RESCO Grid4x4 network, route file,
and configuration copied unchanged. The SUMO configuration refers to files in
the same directory. The upstream network package identifies these assets as
CC BY-NC-ND 4.0; its license text is included as `CC-BY-NC-ND-4.0.pdf`.

Source: [SUMO-RL RESCO Grid4x4](https://github.com/LucasAlegre/sumo-rl/tree/main/sumo_rl/nets/RESCO/grid4x4)
and [RESCO benchmark](https://github.com/Pi-Star-Lab/RESCO).

## 50-seed metrics

Use `src.utils.metrics.TraCIMetricsLogger` to record one completed simulation
step at a time. It reads all controlled incoming lanes and traffic lights from
the active TraCI connection. `run_seed_sweep` runs a rollout callback on seeds
1000–1049 by default and returns per-seed results plus population mean and
standard deviation. `write_seed_sweep_csv` writes both result tables.

The logger reports mean vehicle time loss, maximum observed lane waiting time,
variance of sampled lane mean waiting times, maximum queue length, throughput,
CO₂ emissions (total and g/km), and green-phase switches per hour. The
variance is the population variance over the per-lane mean-wait observations
collected at each simulation step.
