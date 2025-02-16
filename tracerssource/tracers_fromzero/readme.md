# TRACERS (or trace.rs)

## What it do?
Python lib written in rust containing code to do :
- Location trace obfuscation
- Attacker model to denoise the obfuscated traces
- Export the input features for an externally learned attacker model

## Current progress
* obfuscation mechanism works. In overlapping zones it picks the closest EPZ.
* very simple features 

## Priority list attacker model

Training with the currently used features
- Design a loss function
- Export input dataset with rust
- Write pytorch module

Design the emission step
- we need P(observation | all other options) I think. Which is what we have no?
- otherwise reverse engineer the probability... 😞
- use simulated probability 😞
- fit a gaussian 😞
- fit a NN to guess the emission probability 

Better OSM data filtering
- It contains info like trafficlights etc which are currenlty irrelevant and will probably stay irrelevant.
- When getting candidates for a noisy lonlat, it checks if any edge runs close by. If none of these edge points are inside the current threshold, a new node should be generated!!

Better Map Matching
- add pruning during map matching if probability is very low. (this should probably be added somewhere in the rate_trace function, mapping to None trace?)
- add stand still detection, simple idea:
    - preprocess traces such that a subtrace of the same points is mapped to just 2 consecutive identical points.

Add more features:
- time delta between steps
- road type
- don't discard wrong way in a one way street, just make it very high cost.

Complex models
- Stateless NN
- Statefull NN 
- backfeeding RNN / LSTM
- ...

Input partitioning
- Different models for types of users (eg: driving car / walking)


## Priority list obfuscation mechanism
- Low pass mechanism for picking when in overlapping EPZs.


## Ground truth on attacker model:
We want "accurate" traces. We can do this by doing default gps map-matching and checking the error, then defining some treshold to filter "accurate" traces.

Also do some kind of splitting of the traces maybe in case the reporting is only temporarily bad.

Finally filter on some minimum length

This allows for a very automated preprocessing step on the dataset.

Other ground truth source than lifesight: 
- gpx traces of Open Street Maps
- many others.