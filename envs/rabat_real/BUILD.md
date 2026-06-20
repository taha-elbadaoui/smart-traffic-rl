# Real Rabat scenario (OpenStreetMap → SUMO)

Proof-of-concept: a real area of central Rabat (Avenue Mohammed V), imported
from OpenStreetMap. Real roads, real traffic-light junctions, real building
footprints, and plausible synthetic demand.

> **On "real traffic":** the road network, geometry, lane counts, signals and
> buildings are real (from OSM). The *demand* is realistic synthetic traffic
> (`randomTrips.py` + `duarouter`) — there is no open dataset of measured
> vehicle counts for Rabat, so this is the standard SUMO approach.

## How it was built

```bash
# 1. Fetch the OSM extract (bounding box: central Rabat, Av. Mohammed V)
curl "https://api.openstreetmap.org/api/0.6/map?bbox=-6.8395,34.0110,-6.8325,34.0160" -o rabat.osm.xml

# 2. Build the road network (real roads + signals + sidewalks + crossings)
netconvert --osm-files rabat.osm.xml \
  --type-files "$SUMO_HOME/data/typemap/osmNetconvert.typ.xml" \
  -o rabat.net.xml \
  --geometry.remove --roundabouts.guess --ramps.guess \
  --junctions.join --tls.guess-signals --tls.discard-simple --tls.join \
  --keep-edges.by-vclass passenger --no-turnarounds \
  --sidewalks.guess --crossings.guess --walkingareas

# 3. Import real building footprints / parks / water
polyconvert --osm-files rabat.osm.xml --net-file rabat.net.xml \
  --type-file "$SUMO_HOME/data/typemap/osmPolyconvert.typ.xml" \
  -o rabat.poly.xml

# 4. Generate plausible demand and route it
python "$SUMO_HOME/tools/randomTrips.py" -n rabat.net.xml \
  -o rabat.trips.xml -r rabat.rou.xml \
  -e 1000 -p 0.7 --fringe-factor 10 --validate --vehicle-class passenger --prefix veh
```

## View it

```bash
sumo-gui -c envs/rabat_real/rabat.sumocfg
```

To pick a different area, change the `bbox` in step 1 (`left,bottom,right,top`
= `minLon,minLat,maxLon,maxLat`) and re-run.
