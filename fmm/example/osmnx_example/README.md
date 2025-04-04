### Download a routable OSM network in shapefile

Check the download_network.ipynb.

### Run map matching in command line arguments

```bash
# Run precomputation
ubodt_gen --network rome/edges.shp --network_id fid --source u --target v --output rome/ubodt.txt --delta 0.03 --use_omp

# Run fmm
fmm --ubodt rome/ubodt.txt --network rome/edges.shp --network_id fid --source u --target v --gps rome/trips.csv -k 8 -r 0.003 -e 0.0005 --output rome/mr.txt --use_omp

# Run stmatch
stmatch --network rome/edges.shp --network_id fid --source u --target v --gps rome/trips.csv -k 8 -r 0.003 -e 0.0005 --output rome/mr.txt --use_omp --output_fields opath,cpath,mgeom
```

### Run map matching in jupyter notebook

Check the map_matching.ipynb.
