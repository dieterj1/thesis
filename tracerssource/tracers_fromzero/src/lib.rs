// extern crate openblas_src;

use core::f64::consts::PI;
use fixed::prelude::FromFixed;
use fixed::types::I32F32;
use fnv::FnvHasher;
use geo::{
    EuclideanDistance, GeodesicBearing, GeodesicDestination, GeodesicDistance, GeodesicLength,
    HaversineDistance, Point,
};
use itertools::Itertools;
use ndarray::{Array1, Array2};
use ordered_float::OrderedFloat;
use osmpbfreader::OsmObj;
use petgraph::{graph::NodeIndex, prelude::DiGraph, visit::EdgeRef};
use pyo3::prelude::*;

use rand::{seq::SliceRandom, thread_rng};
use rand_distr::{Distribution, Uniform};
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    hash::{BuildHasherDefault, Hash},
    io::Write,
    str::FromStr,
    time::Instant,
};

// defining the road types with a macro for easy extension
macro_rules! define_road_type_enum {
    ($($variant:ident),*) => {
        #[repr(i32)]
        #[derive(Debug, PartialEq, Eq, Copy, Clone)]
        pub enum RoadType {
            $(#[allow(non_camel_case_types)]
            $variant,)*
            Undefined,
        }

        impl std::str::FromStr for RoadType {
            type Err = ();

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $(stringify!($variant) => Ok(RoadType::$variant),)*
                    _ => Ok(RoadType::Undefined),
                }
            }
        }

        impl ToString for RoadType {
            fn to_string(&self) -> String {
                match self {
                    $(RoadType::$variant => String::from(stringify!($variant)),)*
                    RoadType::Undefined => String::from("undefined"),
                }
            }
        }
        impl Default for RoadType {
            fn default() -> Self {
                RoadType::Undefined
            }
        }
        impl IntoPy<PyObject> for RoadType {
            fn into_py(self, py: Python) -> PyObject {
                self.to_string().into_py(py)
            }
        }
    };
}

define_road_type_enum!(
    motorway,
    trunk,
    primary,
    secondary,
    tertiary,
    unclassified,
    residential,
    service,
    motorway_link,
    trunk_link,
    primary_link,
    secondary_link,
    motorway_junction
); // add more road types here

const KMH_TO_MPS: f64 = 1000.0 / 3600.0;

impl RoadType {
    fn to_speed(&self) -> f64 {
        match self {
            // given Rome speed limits, in m/s
            RoadType::motorway => 130.0 * KMH_TO_MPS,
            RoadType::trunk => 90.0 * KMH_TO_MPS,
            RoadType::primary => 70.0 * KMH_TO_MPS,
            RoadType::secondary => 70.0 * KMH_TO_MPS,
            RoadType::tertiary => 50.0 * KMH_TO_MPS,
            RoadType::residential => 30.0 * KMH_TO_MPS,
            RoadType::service => 20.0 * KMH_TO_MPS,
            RoadType::motorway_link => RoadType::motorway.to_speed(),
            RoadType::trunk_link => RoadType::trunk.to_speed(),
            RoadType::primary_link => RoadType::primary.to_speed(),
            RoadType::secondary_link => RoadType::secondary.to_speed(),
            RoadType::motorway_junction => RoadType::motorway.to_speed() / 1.2,
            RoadType::unclassified => 50.0 * KMH_TO_MPS,
            RoadType::Undefined => RoadType::unclassified.to_speed(),
        }
    }
}

// TYPE ALIASSES and STRUCTS
type PrivacyBlob = (f64, f64, f64); // (x,y,radius)
type Settings = (f64, Timestamp); // radius, time interval
type NodeInfo = (i64, f64, f64); // osmid, lon, lat
type Timestamp = u32;
type LonLatTs = (f64, f64, Timestamp); // lon, lat, timestamp
#[derive(Default, Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct Candidate {
    node_index: MyNodeIndex,
    ts: Timestamp,
}
type _FeatureSequence = (Vec<Vec<f64>>, f64);
type PathLonLatTs = Vec<LonLatTs>;

#[derive(Debug, Clone, Copy)]
// (distance, bearing, roadtype)
struct EdgeWeight(f64, f64, RoadType);

impl Default for EdgeWeight {
    fn default() -> Self {
        EdgeWeight(0.0, f64::NAN, RoadType::default())
        // default bearing is NaN because this is used for reflexive edges
    }
}

type NodeIndexType = usize;
type MyNodeIndex = NodeIndex<usize>;
type FastHashMap<U, V> = HashMap<U, V, BuildHasherDefault<FnvHasher>>;

#[derive(Clone, Debug)]
struct State {
    bearing: f64,
    cost: OrderedFloat<f64>,
    backp: Candidate,
    micropath: Vec<Candidate>,
} // currently just a bearing
  // impl new for HiddenState
impl State {
    fn default() -> State {
        State {
            bearing: 0.0f64,
            cost: OrderedFloat(0.0f64),
            backp: Candidate::default(),
            micropath: Vec::default(),
        }
    }
}
// CUSTOM ERROR ENUMS
#[derive(Debug)]
struct PathFindingError;
// START IMPLEMENTS:
#[pyfunction]
// #[pyo3(text_signature = "(settings: (f64, TimestampType), trace: Vec<(f64, f64, TimestampType)>)")]
fn perturb_traces(
    settings: Settings,
    traces: Vec<Vec<(f64, f64, Timestamp)>>,
    chunk_size: Option<usize>,
) -> Vec<Vec<(f64, f64, Timestamp)>> {
    match chunk_size {
        Some(chunk_size) => traces
            .chunks(chunk_size)
            .flat_map(|chunk| {
                chunk
                    .par_iter()
                    .map(|trace| perturb_single_trace(&settings, trace))
                    .collect::<Vec<Vec<(f64, f64, Timestamp)>>>()
            })
            .collect(),
        None => traces
            .par_iter()
            .map(|trace| perturb_single_trace(&settings, trace))
            .collect(),
    }
}

#[pyfunction]
// coordinates have order longutude, latitude!
fn perturb_traces_wsettings(
    _py: Python,
    settings_traces: Vec<(Settings, Vec<(f64, f64, Timestamp)>)>,
    chunk_size: Option<usize>,
) -> Vec<Vec<(f64, f64, Timestamp)>> {
    match chunk_size {
        Some(chunk_size) => settings_traces
            .chunks(chunk_size)
            .flat_map(|chunk| {
                chunk
                    .par_iter()
                    .map(|(settings, trace)| perturb_single_trace(settings, trace))
                    .collect::<Vec<Vec<(f64, f64, Timestamp)>>>()
            })
            .collect(),
        None => settings_traces
            .par_iter()
            .map(|(settings, trace)| perturb_single_trace(settings, trace))
            .collect(),
    }
}

#[pyfunction]
fn split_and_filter_traces(
    traces: Vec<Vec<(f64, f64, Timestamp)>>,
    max_time: Timestamp,
    min_length: i64,
    distance_threshold: f64,
) -> Vec<Vec<(f64, f64, Timestamp)>> {
    let splittraces = split_traces_on_time(traces, max_time);
    // a closure to filter consecutive duplicates
    let is_duplicate = |(lonleft, latleft, _), (lonself, latself, _), (lonright, latright, _)| {
        let leftdist = rs_haversine_distance((lonleft, latleft), (lonself, latself));
        let rightdist = rs_haversine_distance((lonself, latself), (lonright, latright));
        leftdist < distance_threshold && rightdist < distance_threshold
    };
    let traces_no_dup_count: Vec<i64> = splittraces
        .par_iter()
        .map(|trace| {
            let trace_no_dup: Vec<_> = trace
                .windows(3)
                .filter_map(|window| {
                    let (left, zelf, right) = (window[0], window[1], window[2]);
                    if !is_duplicate(left, zelf, right) {
                        Some(zelf)
                    } else {
                        None
                    }
                })
                .collect();
            // count, add 2 because we removed 2 points per trace (start and end)
            (trace_no_dup.len() + 2) as i64
        })
        .collect();
    // zip the traces with the count, filter out counts smaller than min_length, and map to the trace
    splittraces
        .par_iter()
        .zip(traces_no_dup_count)
        .filter_map(|(trace, count)| {
            if count > min_length {
                Some(trace.clone())
            } else {
                None
            }
        })
        .collect()
}

#[pyfunction]
// splitting traces at points where there is too much time in between
fn split_traces_on_time(
    traces: Vec<Vec<(f64, f64, Timestamp)>>,
    max_time: Timestamp,
) -> Vec<Vec<(f64, f64, Timestamp)>> {
    traces
        .par_iter()
        .flat_map(|trace| split_single_trace_on_time(trace, max_time))
        .collect()
}

fn split_single_trace_on_time(
    trace: &Vec<(f64, f64, Timestamp)>,
    max_time: Timestamp,
) -> Vec<Vec<(f64, f64, Timestamp)>> {
    let mut split_traces: Vec<Vec<(f64, f64, Timestamp)>> = Vec::new();
    let mut current_trace: Vec<(f64, f64, Timestamp)> = Vec::new();
    let mut current_timestamp: Timestamp = trace[0].2;
    for (lon, lat, timestamp) in trace {
        if timestamp - current_timestamp > max_time {
            split_traces.push(current_trace);
            current_trace = Vec::new();
        }
        current_trace.push((*lon, *lat, *timestamp));
        current_timestamp = *timestamp;
    }
    split_traces.push(current_trace);
    return split_traces;
}

// Takes: settings, a tuple of (float, int) which is (radius, interval).
// Takes: trace, a vector of tuples of (float, float, int) which is (lon, lat, timestamp).
// Returns: a perturbed vector of tuples of (float, float, int) which is (lon, lat, timestamp).
fn perturb_single_trace(
    settings: &Settings,
    trace: &Vec<(f64, f64, Timestamp)>,
) -> Vec<(f64, f64, Timestamp)> {
    // When we perturb we sample an angle from a uniform distribution between 0 and 2 PI
    let angle_sampler = Uniform::from(0.0..2.0 * PI);
    let mut rng = rand::thread_rng();
    // unpack the settings tuple
    let settings_radius = settings.0;
    let settings_interval = settings.1;
    // The sampled radius is guaranteed to not be larger than the settings radius and uniformly distributed
    // The exact distribution to use is something to optimize (which gives best privacy/utility balance)
    let radius_sampler = Uniform::from(0.0..settings_radius);
    // An array to remember which points have been reported in the past and what area they cover (radius around the point)
    let mut history_blobs: Vec<PrivacyBlob> = Vec::new();
    let tracemapped = trace
        .iter()
        .map(|location_timestamp| {
            let mut choice_location: Point = Point::<f64>::from((0.0, 0.0));
            let mut _choice_delta: (f64, f64) = (0.0, 0.0); // bearing, distance
            let mut choice_timestamp: Timestamp = 0;
            // We use a library called geo to do the geodesic calculations, it has a Point type
            let current_location: Point =
                Point::<f64>::from((location_timestamp.0, location_timestamp.1)); // x:lon, y:lat
            let current_timestamp = location_timestamp.2;
            // if it has not been long enough since the last choice, return the last reported location
            if (current_timestamp - choice_timestamp) < settings_interval {
                // get the longitude from choice_location
                return (choice_location.x(), choice_location.y(), choice_timestamp);
            }
            // check if we have any reports in the past that cover the current location (can be overlapping!)
            let matched_reports: Vec<&PrivacyBlob> = history_blobs
                .par_iter()
                .filter(|report: &&PrivacyBlob| {
                    let pointa: Point = Point::<f64>::from((report.0, report.1)); // x:lon, y:lat
                    let pointb = current_location;
                    // geodesic distance is an accurate model of the earth, returns meters
                    return pointa.geodesic_distance(&pointb) < report.2;
                })
                .collect();
            // if we have reported from within this privacy blob before, make a choice
            // for now we choose the closest privacy blob
            // TODO: make this choice smarter (low pass based on the last choice)
            if matched_reports.len() > 0 {
                let choice: &&PrivacyBlob = matched_reports
                    .iter()
                    // min_by uses the partial_cmp to get the closest matching point in the history
                    .min_by(|reporta, reportb| {
                        #[cfg(debug_assertions)]
                        {
                            println!(
                                "Comparing report a: {:?} with report b: {:?}",
                                reporta, reportb
                            );
                        }
                        let pointa: Point = Point::<f64>::from((reporta.0, reporta.1)); //new(lat,lon)
                        let pointb: Point = Point::<f64>::from((reportb.0, reportb.1)); //new(lat,lon)
                        let da = pointa.geodesic_distance(&current_location);
                        let db = pointb.geodesic_distance(&current_location);
                        return da.partial_cmp(&db).unwrap();
                    })
                    .unwrap();
                let long = choice.0;
                let lat = choice.1;
                #[cfg(debug_assertions)]
                {
                    println!("Chose report: {:?}", choice);
                }
                // to the next iteration we give the reported location, the vector difference to the choice from the current position, and the timestamp
                choice_location = Point::<f64>::from((long, lat));
                // _choice_delta not used currently, but will be useful when we want to make the min_by logic smarter
                _choice_delta = current_location.geodesic_bearing_distance(choice_location);
                choice_timestamp = current_timestamp;
                return (choice_location.x(), choice_location.y(), choice_timestamp);
            } else {
                // this else means we have never been in this area before, and it has been long enough since the last report
                // return the perturbed coordinate
                // prepare a uniform random for the radius and the angle
                let sampled_radius = radius_sampler.sample(&mut rng);
                let sampled_angle = angle_sampler.sample(&mut rng).to_degrees();
                // do the perturbation using teh geodesic earth model
                let newcoord = current_location.geodesic_destination(sampled_angle, sampled_radius);
                // to the next iteration we give the reported location, the vector difference to the choice from the current position, and the timestamp
                choice_location = newcoord;
                _choice_delta = (sampled_angle, sampled_radius);
                choice_timestamp = current_timestamp;
                history_blobs.push((newcoord.x(), newcoord.y(), settings_radius));
                return (choice_location.x(), choice_location.y(), choice_timestamp);
            }
        })
        .collect();
    return tracemapped;
}

// same as the perturb_single_trace but without rayon parallelism, using a single thread
fn perturb_single_trace_single_thread(
    settings: &Settings,
    trace: &Vec<(f64, f64, Timestamp)>,
) -> Vec<(f64, f64, Timestamp)> {
    // When we perturb we sample an angle from a uniform distribution between 0 and 2 PI
    let angle_sampler = Uniform::from(0.0..2.0 * PI);
    let mut rng = rand::thread_rng();
    // unpack the settings tuple
    let settings_radius = settings.0;
    let settings_interval = settings.1;
    // The sampled radius is guaranteed to not be larger than the settings radius and uniformly distributed
    // The exact distribution to use is something to optimize (which gives best privacy/utility balance)
    let radius_sampler = Uniform::from(0.0..settings_radius);
    // An array to remember which points have been reported in the past and what area they cover (radius around the point)
    let mut history_blobs: Vec<PrivacyBlob> = Vec::new();
    let tracemapped = trace
        .iter()
        .map(|location_timestamp| {
            let mut choice_location: Point = Point::<f64>::from((0.0, 0.0));
            let mut _choice_delta: (f64, f64) = (0.0, 0.0); // bearing, distance
            let mut choice_timestamp: Timestamp = 0;
            // We use a library called geo to do the geodesic calculations, it has a Point type
            let current_location: Point =
                Point::<f64>::from((location_timestamp.0, location_timestamp.1)); // x:lon, y:lat
            let current_timestamp = location_timestamp.2;
            // if it has not been long enough since the last choice, return the last reported location
            if (current_timestamp - choice_timestamp) < settings_interval {
                // get the longitude from choice_location
                return (choice_location.x(), choice_location.y(), choice_timestamp);
            }
            // check if we have any reports in the past that cover the current location (can be overlapping!)
            let matched_reports: Vec<&PrivacyBlob> = history_blobs
                .iter()
                .filter(|report: &&PrivacyBlob| {
                    let pointa: Point = Point::<f64>::from((report.0, report.1)); // x:lon, y:lat
                    let pointb = current_location;
                    // geodesic distance is an accurate model of the earth, returns meters
                    return pointa.geodesic_distance(&pointb) < report.2;
                })
                .collect();
            // if we have reported from within this privacy blob before, make a choice
            // for now we choose the closest privacy blob
            // TODO: make this choice smarter (low pass based on the last choice)
            if matched_reports.len() > 0 {
                let choice: &&PrivacyBlob = matched_reports
                    .iter()
                    // min_by uses the partial_cmp to get the closest matching point in the history
                    .min_by(|reporta, reportb| {
                        #[cfg(debug_assertions)]
                        {
                            println!(
                                "Comparing report a: {:?} with report b: {:?}",
                                reporta, reportb
                            );
                        }
                        let pointa: Point = Point::<f64>::from((reporta.0, reporta.1)); //new(lat,lon)
                        let pointb: Point = Point::<f64>::from((reportb.0, reportb.1)); //new(lat,lon)
                        let da = pointa.geodesic_distance(&current_location);
                        let db = pointb.geodesic_distance(&current_location);
                        return da.partial_cmp(&db).unwrap();
                    })
                    .unwrap();
                let long = choice.0;
                let lat = choice.1;
                #[cfg(debug_assertions)]
                {
                    println!("Chose report: {:?}", choice);
                }
                // to the next iteration we give the reported location, the vector difference to the choice from the current position, and the timestamp
                choice_location = Point::<f64>::from((long, lat));
                // _choice_delta not used currently, but will be useful when we want to make the min_by logic smarter
                _choice_delta = current_location.geodesic_bearing_distance(choice_location);
                choice_timestamp = current_timestamp;
                return (choice_location.x(), choice_location.y(), choice_timestamp);
            } else {
                // this else means we have never been in this area before, and it has been long enough since the last report
                // return the perturbed coordinate
                // prepare a uniform random for the radius and the angle
                let sampled_radius = radius_sampler.sample(&mut rng);
                let sampled_angle = angle_sampler.sample(&mut rng).to_degrees();
                // do the perturbation using teh geodesic earth model
                let newcoord = current_location.geodesic_destination(sampled_angle, sampled_radius);
                // to the next iteration we give the reported location, the vector difference to the choice from the current position, and the timestamp
                choice_location = newcoord;
                _choice_delta = (sampled_angle, sampled_radius);
                choice_timestamp = current_timestamp;
                history_blobs.push((newcoord.x(), newcoord.y(), settings_radius));
                return (choice_location.x(), choice_location.y(), choice_timestamp);
            }
        })
        .collect();
    return tracemapped;
}

#[pyfunction]
fn perturb_traces_wpool(
    settings: Settings,
    traces: Vec<Vec<(f64, f64, Timestamp)>>,
    pool_size: Option<usize>,
) -> Vec<Vec<(f64, f64, Timestamp)>> {
    let pool_size = match pool_size {
        None => 72,
        Some(pool_size) => pool_size,
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(pool_size)
        .build()
        .expect("Error in building thread pool");
    pool.install(|| {
        traces
            .par_iter()
            .map(|trace| perturb_single_trace_single_thread(&settings, trace))
            .collect()
    })
}

#[pyfunction]
fn get_path_length_lonlat(lonlats: Vec<(f64, f64)>) -> f64 {
    let linestringlonlats = geo::LineString::from(lonlats);
    linestringlonlats.geodesic_length()
}

#[pyclass]
#[derive(Debug)]
struct PyGraph {
    graph: DiGraph<NodeInfo, EdgeWeight, usize>, //this defines the NodeIndex used in the graph internal type as usize
    osmid_to_nodeindex: HashMap<i64, MyNodeIndex>,
    #[pyo3(get, set)]
    pub yendepth: usize,
    pub ml_weights: Array1<f64>,
}

#[pymethods]
impl PyGraph {
    #[new]
    fn new() -> Self {
        let graph = DiGraph::<NodeInfo, EdgeWeight, usize>::default();
        let osmid_to_nodeindex = HashMap::new();
        PyGraph {
            graph,
            osmid_to_nodeindex,
            yendepth: 2,
            ml_weights: vec![1.0, 0.01, 0.0].into(),
        }
    }
    #[getter]
    fn get_ml_weights(&self) -> Vec<f64> {
        self.ml_weights.to_vec()
    }
    #[setter]
    fn set_ml_weights(&mut self, weights: Vec<f64>) {
        self.ml_weights = Array1::from(weights);
    }
    // return osmid, lon, lat
    fn get_node_weight(&self, nodeid: usize) -> Option<(i64, f64, f64)> {
        let node_weight = self.graph.node_weight(nodeid.into())?;
        return Some(*node_weight);
    }

    fn osm_to_nodeid(&self, osmid: i64) -> Option<usize> {
        let nodeid = self.osmid_to_nodeindex.get(&osmid)?;
        return Some(nodeid.index());
    }

    // bounding_box is minlon, maxlon, minlat, maxlat
    fn load_graph_base(
        &mut self,
        filepath: &str,
        roadtype_selection: Vec<&str>,
        bounding_box: Option<(f64, f64, f64, f64)>,
        check_oneway: Option<bool>,
    ) {
        *self = graph_from_file(filepath, roadtype_selection, bounding_box, check_oneway)
    }

    fn load_graph(&mut self, filepath: &str) {
        *self = graph_from_file(
            filepath,
            vec![
                "motorway",
                "trunk",
                "primary",
                "secondary",
                "tertiary",
                "residential",
                "unclassified",
            ],
            None,
            None,
        )
    }

    fn load_graph_all(&mut self, filepath: &str) {
        *self = graph_from_file(filepath, vec![], None, Some(true))
    }

    fn map_match_multiple_traces(
        &self,
        traces: Vec<Vec<LonLatTs>>,
        radius: f64,
        outfileprefix: &str,
    ) {
        traces
            .into_par_iter()
            .enumerate()
            .for_each(|(numi, trace)| {
                let outfilename = format!("{}_{}.csv", outfileprefix, numi);
                // clear file if exists
                std::fs::write(&outfilename, "").expect("Error in writing file");
                // open in append mode
                let mut file = std::fs::OpenOptions::new()
                    .append(true)
                    .open(&outfilename)
                    .expect("Error in opening file");
                let (og_result_splits, mmresult) = self.map_match_trace(trace, radius);
                mmresult.into_iter().zip(og_result_splits.iter()).for_each(
                    |(mmresult, og_result)| {
                        // first microresult is an empty vec, as we don't know how to arrive at the first OG point
                        let trace_str = match mmresult {
                            Some(some_trace) => some_trace
                                .iter()
                                .map(|(index, ts)| {
                                    let (osmid, lon, lat) = self.get_node_weight(*index).unwrap();
                                    format!("{},{},{},{}", osmid, lon, lat, ts)
                                })
                                .collect::<Vec<String>>()
                                .join("|"),
                            None => "".to_string(),
                        };
                        // write trace_str to file
                        file.write_all(trace_str.as_bytes())
                            .expect("Error in writing file");
                        file.write_all("\n".as_bytes())
                            .expect("Error in writing file");
                        let og_trace_str =
                            format!("{},{},{}", og_result.0, og_result.1, og_result.2);
                        file.write_all(og_trace_str.as_bytes())
                            .expect("Error in writing file");
                        file.write_all("\n".as_bytes())
                            .expect("Error in writing file");
                    },
                )
            });
    }

    // input is list of latlon (+ts in future), returns list of nodeids
    fn map_match_trace(
        &self,
        input_trace: Vec<LonLatTs>,
        radius: f64,
    ) -> (Vec<LonLatTs>, Vec<Option<Vec<(NodeIndexType, Timestamp)>>>) {
        //option is for when no path is found
        // check if load_graph has been called by checking if osmid_to_nodeindex is empty
        if self.osmid_to_nodeindex.is_empty() {
            panic!("load_graph has not been called");
        }

        let mut split_indices: Vec<usize> = vec![0];

        let candidates: Vec<Vec<Candidate>> =
            self.candidate_ids_ts(input_trace.clone(), radius * 1.1f64);
        //add 10% for noise and edges crossing through the area but having no nodes in it

        // hidden state tracks how we got to a node, the cost is the sum of the costs of the previous nodes, best we could find.
        // backp is the previous node in the best path
        // micropath is the path from the previous node to the current node
        let mut state_t_maps: Vec<Option<HashMap<Candidate, State>>> = candidates
            .par_iter()
            .map(|clist| Some(clist.iter().map(|&cand| (cand, State::default())).collect()))
            .collect();
        let default_state_t_maps = state_t_maps.clone();

        for t in 1..candidates.len() {
            let (left, right) = state_t_maps.split_at_mut(t);
            let prev_states = match left.last().unwrap().as_ref() {
                Some(prev_state) => prev_state,
                None => default_state_t_maps[t - 1].as_ref().unwrap(),
            };
            let curr_hidden_states = right.first_mut().unwrap();
            let curr_candidates = &candidates[t];
            let unfiltered_curr_hidden_states: HashMap<Candidate, State> = curr_candidates
                .into_par_iter()
                .filter_map(|cand| {
                    let for_previd_best_path_and_state: Option<State> = prev_states
                        .iter()
                        .filter_map(|(&prev_cand, prev_state)| {
                            match self.prev_to_curr_cost(
                                &prev_cand,
                                cand,
                                prev_states,
                                PyGraph::single_path_cost,
                            ) {
                                Some(mut path_state) => {
                                    path_state.cost += prev_state.cost;
                                    return Some(path_state);
                                }
                                None => None,
                            }
                        })
                        .min_by_key(|thestate| thestate.cost);
                    match for_previd_best_path_and_state {
                        Some(beststate) => Some((cand.clone(), beststate)),
                        None => None,
                    }
                })
                .collect();
            *curr_hidden_states = match unfiltered_curr_hidden_states.len() {
                0 => {
                    println!(
                        "There were no valid states passed from {} to {} \n state was set to None",
                        t - 1,
                        t
                    );
                    split_indices.push(t);
                    None
                }
                _ => Some(unfiltered_curr_hidden_states),
            };
        }

        let mut microresult: Vec<Option<Vec<Candidate>>> = vec![];
        // this will backtrace the best path starting from the last (valid) state with the lowest cost
        let mut cand_iterator: Option<Candidate> = None;
        for statemap_t in state_t_maps.iter().rev() {
            // if statemap_t is None, we set state_iterator to None again and push None to microresult
            match statemap_t {
                None => {
                    // a split is found, reset the state_iterator
                    cand_iterator = None;
                    microresult.push(None);
                }
                Some(statemap) => {
                    match cand_iterator {
                        None => {
                            // we havent found any backpointers to backtrack yet, so we find the one with the lowest cost
                            cand_iterator = statemap
                                .iter()
                                .min_by_key(|(_, state)| state.cost)
                                .map(|(cand, _state)| cand.clone());
                            // assert not none now
                            assert!(cand_iterator.is_some(), "statemap was not None, but no min was found, should also not be empty ...");
                        }
                        Some(_) => {}
                    }
                    let best_state_t = statemap.get(&cand_iterator.unwrap()).unwrap_or_else(|| {
                        panic!("Error in backtracking micropaths");
                    });
                    microresult.push(Some(best_state_t.micropath.clone()));
                    // update the state_iterator to the backpointer of the current state
                    cand_iterator = Some(best_state_t.backp.clone());
                }
            }
        }
        assert_eq!(
            input_trace.len(),
            microresult.len(),
            "input trace and microresult should have the same length"
        );

        return (
            input_trace,
            microresult
                .into_iter()
                .rev()
                .map(|opt_vec| {
                    opt_vec.map(|vec| {
                        vec.into_iter()
                            .map(|cand| (cand.node_index.index(), cand.ts))
                            .collect()
                    })
                })
                .collect(),
        );
    }

    fn reports_sample_features(
        &self,
        emitted_trace_lonlats: Vec<LonLatTs>,
        real_trace_lonlat: Vec<Vec<(f64, f64)>>, //vec per timestamp (transition)
        radius: f64,
        try_take_amount: Option<usize>,
        timeout: Option<usize>,
    ) -> Vec<(Vec<Vec<f64>>, f64)> {
        let try_take_amount = match try_take_amount {
            None => usize::MAX,
            Some(amount) => amount,
        };
        let t0 = std::time::Instant::now();
        let mut cand_t_list: Vec<_> = self.candidate_ids_ts(emitted_trace_lonlats.clone(), radius);
        // do a checked multiplication of the lens of the vecs inside reduced_cand_t_list
        let total_combos = match cand_t_list
            .iter()
            .map(|clist| clist.len())
            .try_fold(1usize, |acc, x| acc.checked_mul(x))
        {
            Some(total) => total,
            None => {
                let large_number = u32::MAX;
                // calc the cand.len() log of large number
                let log_large_number = (large_number as f64).log(cand_t_list.len() as f64) as usize;

                cand_t_list = cand_t_list
                    .into_iter()
                    .map(|clist| clist[..log_large_number].to_vec())
                    .collect_vec();
                cand_t_list
                    .iter()
                    .map(|clist| clist.len())
                    .try_fold(1usize, |acc, x| acc.checked_mul(x))
                    .expect("Error in calculating total_combos")
            }
        };
        let candidate_combinations = cand_t_list
            .into_iter()
            .multi_cartesian_product()
            .enumerate()
            .filter(|(i, _)| {
                (*i as f64 % (total_combos as f64 / try_take_amount as f64)).abs() < 1.0
            })
            .take(try_take_amount)
            .map(|(_, cand_combo_t)| cand_combo_t);
        // for each single combo_t
        let pairs_all_combinations =
            candidate_combinations
                .par_bridge()
                .map(|candidate_combination_t| {
                    let optional_path_combo_t = candidate_combination_t
                        // zip with skip(1) to get the pairs of candidates
                        .iter()
                        .zip(candidate_combination_t.iter().skip(1))
                        .map(|(&a, &b)| (a, b))
                        .collect_vec();
                    optional_path_combo_t
                });

        // fold into hashmap<pair.0, vec<cand>> aka hashmap<startnode, vec<endnode>>
        let map_pairs: FastHashMap<Candidate, HashSet<Candidate>> = pairs_all_combinations
            .clone()
            .fold(
                || HashMap::default(),
                |mut acc, pair_iter| {
                    for (mut a, mut b) in pair_iter.into_iter() {
                        b.ts -= a.ts;
                        a.ts = 0;
                        acc.entry(a).or_insert_with(HashSet::new).insert(b);
                    }
                    acc
                },
            )
            .reduce(
                || HashMap::default(),
                |mut acc, local| {
                    for (key, value) in local {
                        acc.entry(key).or_insert_with(HashSet::new).extend(value);
                    }
                    acc
                },
            );
        // call multi_yen(startnode, vec<endnode>, yendepth) and put in antoher hashmap<(startnode, endnode), vec<paths>>
        let path_cache: FastHashMap<_, _> = map_pairs
            .into_par_iter()
            .map(|(startnode, endnodes)| {
                // TODO: if stepping forward through time, we should only calculate for sources at time t if they are reachable from t-1
                // although this is weird to optimize as the pairs are flattened over time
                let paths = self.multi_yen(&startnode, &endnodes, self.yendepth);
                (startnode.node_index, paths)
            })
            .fold(
                || FastHashMap::default(),
                |mut acc, start_target_paths_vec| {
                    let (startnode, target_paths_vec) = start_target_paths_vec;
                    for (target, path) in target_paths_vec {
                        acc.entry((startnode, target))
                            .or_insert_with(Vec::new)
                            .push(path);
                    }
                    acc
                },
            )
            .reduce(
                || HashMap::default(),
                |mut acc, local| {
                    for (key, value) in local {
                        acc.entry(key).or_insert_with(Vec::new).extend(value);
                    }
                    acc
                },
            );
        //redo the multi_cartesian_product
        // + pair forming (should be deterministic, otherwise collect into vec and clone) --> this vec is pairs_all_combinations
        let map_errors_to_feat_vecs : FastHashMap<_,_>= pairs_all_combinations.flat_map_iter(|pairs_one_combo_t| {
            // call then cache<(startnode, endnode), vec<paths>>.get() for the pair, shortcircuit if not found
            let multi_edge_t: Vec<_> = pairs_one_combo_t.iter()
                .map(|(a, b)| {
                    let key = (a, b);
                    let edge = path_cache.get(&(key.0.node_index, key.1.node_index)); //TODO: short circuit if not found
                    match edge {
                        Some(paths) => (*paths).clone(),
                        None => vec![],
                    }
                })
                .collect();
            let all_edge_combos = multi_edge_t.into_iter().multi_cartesian_product();

            let all_errors_and_featr = all_edge_combos.filter_map(|edges_t| {

                let full_path_nodes: Vec<MyNodeIndex> = std::iter::once(edges_t[0][0].clone())
                    .chain(edges_t.iter().flat_map(|path| path[1..].to_vec()))
                    .collect();
                // assert edges t not empty
                if full_path_nodes.is_empty(){
                    println!("second multicartesian produced an empty vec (which means one timestamp had no path)");
                    return None;
                }
                let _full_path: Vec<(f64, f64, u32, i64)> = full_path_nodes.iter().map(|node| {
                    let (osmid, lon, lat) = self.get_node_weight(node.index()).unwrap();
                    return (lon, lat, 0, osmid);
                }).collect();
                let real_lonlat = std::iter::once(real_trace_lonlat[0][0].clone())
                    .chain(real_trace_lonlat.iter().flat_map(|path| path[1..].to_vec()))
                    .collect();
                let full_path_lonlat = full_path_nodes
                    .iter()
                    .map(|node| {
                        let (_osmid, lon, lat) = self.get_node_weight(node.index()).unwrap();
                        return (lon, lat);
                    })
                    .collect_vec();
                let error = prediction_error_base(&full_path_lonlat, &real_lonlat);
                let one_seq_context = full_path_nodes
                    .clone()
                    .into_iter()
                    .zip(full_path_nodes.into_iter().skip(1))
                    .zip(
                        real_lonlat
                            .iter()
                            .cloned()
                            .zip(real_lonlat.iter().cloned().skip(1)),
                    )
                    .map(|((a, b), ((c, d), (e, f)))| (a, b, c, d, e, f))
                    .collect_vec();
                Some((error.unwrap(), one_seq_context))
            });
            all_errors_and_featr
        })
        .take_any_while(|_|{
            timeout.unwrap_or(usize::MAX) > (Instant::now() - t0).as_secs() as usize
        })
        // fold into hashmap<error, vec<fullpath>>
        .fold(|| HashMap::default(), |mut acc, (error, full_path)| {
            acc.insert(error, full_path); // only keep the first path with the same error
            acc
        }).into_par_iter()
        .reduce(|| HashMap::default(), |mut acc, local| {
            for (key, value) in local {
                acc.insert(key, value);
            }
            acc
        });
        let sampled_features = map_errors_to_feat_vecs
            .into_iter()
            .map(|(error, full_path)| {
                let featr = full_path
                    .into_iter()
                    .map(|(a, b, ealon, ealat, eblon, eblat)| {
                        self.get_transition_features(
                            a,
                            b,
                            Point::<f64>::from((ealon, ealat)),
                            Point::<f64>::from((eblon, eblat)),
                        )
                    })
                    .collect_vec();
                (featr, FromFixed::from_fixed(error))
            })
            .collect();
        sampled_features
    }

    fn reports_sample_features_and_paths(
        &self,
        emitted_trace_lonlats: Vec<LonLatTs>,
        real_trace_lonlat: Vec<Vec<(f64, f64)>>, //vec per timestamp (transition)
        radius: f64,
        try_take_amount: Option<usize>,
        timeout: Option<usize>,
    ) -> Vec<(PathLonLatTs, Vec<Vec<f64>>, f64)> {
        let try_take_amount = match try_take_amount {
            None => usize::MAX,
            Some(amount) => amount,
        };
        let t0 = std::time::Instant::now();
        let mut cand_t_list: Vec<_> = self.candidate_ids_ts(emitted_trace_lonlats.clone(), radius);
        // do a checked multiplication of the lens of the vecs inside reduced_cand_t_list
        let total_combos = match cand_t_list
            .iter()
            .map(|clist| clist.len())
            .try_fold(1usize, |acc, x| acc.checked_mul(x))
        {
            Some(total) => total,
            None => {
                let large_number = u32::MAX;
                // calc the cand.len() log of large number
                let log_large_number = (large_number as f64).log(cand_t_list.len() as f64) as usize;

                cand_t_list = cand_t_list
                    .into_iter()
                    .map(|clist| clist[..log_large_number].to_vec())
                    .collect_vec();
                cand_t_list
                    .iter()
                    .map(|clist| clist.len())
                    .try_fold(1usize, |acc, x| acc.checked_mul(x))
                    .expect("Error in calculating total_combos")
            }
        };
        let candidate_combinations = cand_t_list
            .into_iter()
            .multi_cartesian_product()
            .enumerate()
            .filter(|(i, _)| {
                (*i as f64 % (total_combos as f64 / try_take_amount as f64)).abs() < 1.0
            })
            .take(try_take_amount)
            .map(|(_, cand_combo_t)| cand_combo_t);
        // for each single combo_t
        let pairs_all_combinations =
            candidate_combinations
                .par_bridge()
                .map(|candidate_combination_t| {
                    let optional_path_combo_t = candidate_combination_t
                        // zip with skip(1) to get the pairs of candidates
                        .iter()
                        .zip(candidate_combination_t.iter().skip(1))
                        .map(|(&a, &b)| (a, b))
                        .collect_vec();
                    optional_path_combo_t
                });

        // fold into hashmap<pair.0, vec<cand>> aka hashmap<startnode, vec<endnode>>
        let map_pairs: FastHashMap<Candidate, HashSet<Candidate>> = pairs_all_combinations
            .clone()
            .fold(
                || HashMap::default(),
                |mut acc, pair_iter| {
                    for (mut a, mut b) in pair_iter.into_iter() {
                        b.ts -= a.ts;
                        a.ts = 0;
                        acc.entry(a).or_insert_with(HashSet::new).insert(b);
                    }
                    acc
                },
            )
            .reduce(
                || HashMap::default(),
                |mut acc, local| {
                    for (key, value) in local {
                        acc.entry(key).or_insert_with(HashSet::new).extend(value);
                    }
                    acc
                },
            );
        // call multi_yen(startnode, vec<endnode>, yendepth) and put in antoher hashmap<(startnode, endnode), vec<paths>>
        let path_cache: FastHashMap<_, _> = map_pairs
            .into_par_iter()
            .map(|(startnode, endnodes)| {
                // TODO: if stepping forward through time, we should only calculate for sources at time t if they are reachable from t-1
                // although this is weird to optimize as the pairs are flattened over time
                let paths = self.multi_yen(&startnode, &endnodes, self.yendepth);
                (startnode.node_index, paths)
            })
            .fold(
                || FastHashMap::default(),
                |mut acc, start_target_paths_vec| {
                    let (startnode, target_paths_vec) = start_target_paths_vec;
                    for (target, path) in target_paths_vec {
                        acc.entry((startnode, target))
                            .or_insert_with(Vec::new)
                            .push(path);
                    }
                    acc
                },
            )
            .reduce(
                || HashMap::default(),
                |mut acc, local| {
                    for (key, value) in local {
                        acc.entry(key).or_insert_with(Vec::new).extend(value);
                    }
                    acc
                },
            );
        //redo the multi_cartesian_product
        // + pair forming (should be deterministic, otherwise collect into vec and clone) --> this vec is pairs_all_combinations
        let map_errors_to_feat_vecs : FastHashMap<_,_>= pairs_all_combinations.flat_map_iter(|pairs_one_combo_t| {
            // call then cache<(startnode, endnode), vec<paths>>.get() for the pair, shortcircuit if not found
            let multi_edge_t: Vec<_> = pairs_one_combo_t.iter()
                .map(|(a, b)| {
                    let key = (a, b);
                    let edge = path_cache.get(&(key.0.node_index, key.1.node_index)); //TODO: short circuit if not found
                    match edge {
                        Some(paths) => (*paths).clone(),
                        None => vec![],
                    }
                })
                .collect();
            let all_edge_combos = multi_edge_t.into_iter().multi_cartesian_product();

            let all_errors_and_featr = all_edge_combos.filter_map(|edges_t| {

                let full_path_nodes: Vec<MyNodeIndex> = std::iter::once(edges_t[0][0].clone())
                    .chain(edges_t.iter().flat_map(|path| path[1..].to_vec()))
                    .collect();
                // assert edges t not empty
                if full_path_nodes.is_empty(){
                    println!("second multicartesian produced an empty vec (which means one timestamp had no path)");
                    return None;
                }
                let full_path: Vec<(f64, f64, i64)> = full_path_nodes.iter().map(|node| {
                    let (osmid, lon, lat) = self.get_node_weight(node.index()).unwrap();
                    return (lon, lat, osmid);
                }).collect();
                let real_lonlat = std::iter::once(real_trace_lonlat[0][0].clone())
                    .chain(real_trace_lonlat.iter().flat_map(|path| path[1..].to_vec()))
                    .collect();
                let full_path_lonlat = full_path_nodes
                    .iter()
                    .map(|node| {
                        let (_osmid, lon, lat) = self.get_node_weight(node.index()).unwrap();
                        return (lon, lat);
                    })
                    .collect_vec();
                let error = prediction_error_base(&full_path_lonlat, &real_lonlat);
                let one_seq_context = full_path_nodes
                    .clone()
                    .into_iter()
                    .zip(full_path_nodes.into_iter().skip(1))
                    .zip(
                        real_lonlat
                            .iter()
                            .cloned()
                            .zip(real_lonlat.iter().cloned().skip(1)),
                    )
                    .map(|((a, b), ((c, d), (e, f)))| (a, b, c, d, e, f))
                    .collect_vec();
                Some((error.unwrap(), (one_seq_context, full_path)))
            });
            all_errors_and_featr
        })
        .take_any_while(|_|{
            timeout.unwrap_or(usize::MAX) > (Instant::now() - t0).as_secs() as usize
        })
        // fold into hashmap<error, vec<fullpath>>
        .fold(|| HashMap::default(), |mut acc, (error, context_and_path)| {
            acc.insert(error, context_and_path); // only keep the first path with the same error
            acc
        }).into_par_iter()
        .reduce(|| HashMap::default(), |mut acc, local| {
            for (key, value) in local {
                acc.insert(key, value);
            }
            acc
        });
        let sampled_features = map_errors_to_feat_vecs
            .into_iter()
            .map(|(error, context_and_path)| {
                let (context, full_path) = context_and_path;
                let path = full_path
                    .iter()
                    .map(|(lon, lat, _osmid)| {
                        return (*lon, *lat, 0);
                    })
                    .collect_vec();
                let featr = context
                    .into_iter()
                    .map(|(a, b, ealon, ealat, eblon, eblat)| {
                        self.get_transition_features(
                            a,
                            b,
                            Point::<f64>::from((ealon, ealat)),
                            Point::<f64>::from((eblon, eblat)),
                        )
                    })
                    .collect_vec();
                (path, featr, FromFixed::from_fixed(error))
            })
            .collect();
        sampled_features
    }

    fn reports_sample_paths(
        &self,
        emitted_trace_lonlats: Vec<LonLatTs>,
        real_trace_lonlat: Vec<Vec<(f64, f64)>>, //vec per timestamp (transition)
        radius: f64,
        try_take_amount: Option<usize>,
        timeout: Option<usize>,
    ) -> Vec<(PathLonLatTs, f64)> {
        let try_take_amount = match try_take_amount {
            None => usize::MAX,
            Some(amount) => amount,
        };
        let t0 = std::time::Instant::now();
        let mut cand_t_list: Vec<_> = self.candidate_ids_ts(emitted_trace_lonlats.clone(), radius);
        // do a checked multiplication of the lens of the vecs inside reduced_cand_t_list
        let total_combos = match cand_t_list
            .iter()
            .map(|clist| clist.len())
            .try_fold(1usize, |acc, x| acc.checked_mul(x))
        {
            Some(total) => total,
            None => {
                let large_number = u32::MAX;
                // calc the cand.len() log of large number
                let log_large_number = (large_number as f64).log(cand_t_list.len() as f64) as usize;

                cand_t_list = cand_t_list
                    .into_iter()
                    .map(|clist| clist[..log_large_number].to_vec())
                    .collect_vec();
                cand_t_list
                    .iter()
                    .map(|clist| clist.len())
                    .try_fold(1usize, |acc, x| acc.checked_mul(x))
                    .expect("Error in calculating total_combos")
            }
        };
        let candidate_combinations = cand_t_list
            .into_iter()
            .multi_cartesian_product()
            .enumerate()
            .filter(|(i, _)| {
                (*i as f64 % (total_combos as f64 / try_take_amount as f64)).abs() < 1.0
            })
            .take(try_take_amount)
            .map(|(_, cand_combo_t)| cand_combo_t);
        // for each single combo_t
        let pairs_all_combinations =
            candidate_combinations
                .par_bridge()
                .map(|candidate_combination_t| {
                    let optional_path_combo_t = candidate_combination_t
                        // zip with skip(1) to get the pairs of candidates
                        .iter()
                        .zip(candidate_combination_t.iter().skip(1))
                        .map(|(&a, &b)| (a, b))
                        .collect_vec();
                    optional_path_combo_t
                });

        // fold into hashmap<pair.0, vec<cand>> aka hashmap<startnode, vec<endnode>>
        let map_pairs: FastHashMap<Candidate, HashSet<Candidate>> = pairs_all_combinations
            .clone()
            .fold(
                || HashMap::default(),
                |mut acc, pair_iter| {
                    for (mut a, mut b) in pair_iter.into_iter() {
                        b.ts -= a.ts;
                        a.ts = 0;
                        acc.entry(a).or_insert_with(HashSet::new).insert(b);
                    }
                    acc
                },
            )
            .reduce(
                || HashMap::default(),
                |mut acc, local| {
                    for (key, value) in local {
                        acc.entry(key).or_insert_with(HashSet::new).extend(value);
                    }
                    acc
                },
            );
        // call multi_yen(startnode, vec<endnode>, yendepth) and put in antoher hashmap<(startnode, endnode), vec<paths>>
        let path_cache: FastHashMap<_, _> = map_pairs
            .into_par_iter()
            .map(|(startnode, endnodes)| {
                // TODO: if stepping forward through time, we should only calculate for sources at time t if they are reachable from t-1
                // although this is weird to optimize as the pairs are flattened over time
                let paths = self.multi_yen(&startnode, &endnodes, self.yendepth);
                (startnode.node_index, paths)
            })
            .fold(
                || FastHashMap::default(),
                |mut acc, start_target_paths_vec| {
                    let (startnode, target_paths_vec) = start_target_paths_vec;
                    for (target, path) in target_paths_vec {
                        acc.entry((startnode, target))
                            .or_insert_with(Vec::new)
                            .push(path);
                    }
                    acc
                },
            )
            .reduce(
                || HashMap::default(),
                |mut acc, local| {
                    for (key, value) in local {
                        acc.entry(key).or_insert_with(Vec::new).extend(value);
                    }
                    acc
                },
            );
        //redo the multi_cartesian_product
        // + pair forming (should be deterministic, otherwise collect into vec and clone) --> this vec is pairs_all_combinations
        let map_errors_to_feat_vecs : FastHashMap<_,_>= pairs_all_combinations.flat_map_iter(|pairs_one_combo_t| {
            // call then cache<(startnode, endnode), vec<paths>>.get() for the pair, shortcircuit if not found
            let multi_edge_t: Vec<_> = pairs_one_combo_t.iter()
                .map(|(a, b)| {
                    let key = (a, b);
                    let edge = path_cache.get(&(key.0.node_index, key.1.node_index)); //TODO: short circuit if not found
                    match edge {
                        Some(paths) => (*paths).clone(),
                        None => vec![],
                    }
                })
                .collect();
            let all_edge_combos = multi_edge_t.into_iter().multi_cartesian_product();

            let all_errors_and_paths = all_edge_combos.filter_map(|edges_t| {

                let full_path: Vec<MyNodeIndex> = std::iter::once(edges_t[0][0].clone())
                    .chain(edges_t.iter().flat_map(|path| path[1..].to_vec()))
                    .collect();
                // assert edges t not empty
                if full_path.is_empty(){
                    println!("second multicartesian produced an empty vec (which means one timestamp had no path)");
                    return None;
                }
                let real_lonlat = std::iter::once(real_trace_lonlat[0][0].clone())
                    .chain(real_trace_lonlat.iter().flat_map(|path| path[1..].to_vec()))
                    .collect();
                let full_path_lonlat = full_path
                    .iter()
                    .map(|node| {
                        let (_osmid, lon, lat) = self.get_node_weight(node.index()).unwrap();
                        return (lon, lat);
                    })
                    .collect_vec();
                let error = prediction_error_base(&full_path_lonlat, &real_lonlat);
                Some((error.unwrap(), full_path))
            });
            all_errors_and_paths
        })
        .take_any_while(|_|{
            timeout.unwrap_or(usize::MAX) > (Instant::now() - t0).as_secs() as usize
        })
        // fold into hashmap<error, vec<fullpath>>
        .fold(|| HashMap::default(), |mut acc, (error, full_path)| {
            acc.insert(error, full_path); // only keep the first path with the same error
            acc
        }).into_par_iter()
        .reduce(|| HashMap::default(), |mut acc, local| {
            for (key, value) in local {
                acc.insert(key, value);
            }
            acc
        });
        map_errors_to_feat_vecs
            .into_iter()
            .map(|(error, full_path)| {
                let path = full_path
                    .iter()
                    .map(|node| {
                        let (_osmid, lon, lat) = self.get_node_weight(node.index()).unwrap();
                        return (lon, lat, 0);
                    })
                    .collect_vec();
                (path, FromFixed::from_fixed(error))
            })
            .collect()
    }

    // we pass in the perturbed lonlats, and the real trace ids.
    // we give back the features of all choices the given id trace made within the context of the perturbed trace (emits)
    // this is used for training the ML model
    // see the get_transition_features function to see what features are extracted
    fn extract_trace_transition_features(
        &self,
        stacked_groundtruth_ids: Vec<Vec<NodeIndexType>>,
        emitted_trace_lonlats: Vec<(f64, f64)>,
    ) -> Vec<Vec<f64>> {
        assert_eq!(
            stacked_groundtruth_ids.len(),
            emitted_trace_lonlats.len() - 1
        );
        // this gives features for a single vec of path combinations between the reports
        // convert nodeids to MyNodeIndex and flatten
        // the last groundthruth_id in each subvec is the first groundtruth_id in the next subvec
        vec![vec![
            usize::default(),
            stacked_groundtruth_ids[0][0].clone(),
        ]]
        .into_iter()
        .chain(stacked_groundtruth_ids.into_iter())
        .zip(emitted_trace_lonlats.clone().into_iter())
        .zip(emitted_trace_lonlats.into_iter().skip(1))
        .flat_map(|((trace_edge, emit_a), emit_b)| {
            trace_edge[1..]
                .iter()
                .zip(trace_edge[2..].iter())
                .map(|(&nodeid, &nodeid_next)| {
                    let osm_a = NodeIndex::new(nodeid);
                    let osm_b = NodeIndex::new(nodeid_next);
                    let emita_point = Point::<f64>::from(emit_a);
                    let emitb_point = Point::<f64>::from(emit_b);
                    self.get_transition_features(osm_a, osm_b, emita_point, emitb_point)
                        .clone()
                })
                .collect::<Vec<_>>()
        })
        .collect()
    }

    fn get_path_length_nodeid(&self, osmids: Vec<NodeIndexType>) -> f64 {
        return get_path_length_lonlat(
            osmids
                .par_iter()
                .map(|&nodeid| {
                    let (_, lon, lat) = self
                        .get_node_weight(nodeid)
                        .expect("Error in getting node weight");
                    return (lon, lat);
                })
                .collect(),
        );
    }
}

// here go pure rust functions not exposed to python
// we are getting feature at each microtransition from real_node to next_real_node
// input is IDS not osmids
impl PyGraph {
    // This gives features for a single step of the (micro) path
    fn get_transition_features(
        &self,
        real_node: MyNodeIndex,
        next_real_node: MyNodeIndex,
        emmit_a: Point<f64>,
        emmit_b: Point<f64>,
    ) -> Vec<f64> {
        let real_weight = self.graph.node_weight(real_node).unwrap();
        let real_point = Point::<f64>::from((real_weight.1, real_weight.2));

        let edge_index = self
            .graph
            .find_edge(real_node, next_real_node)
            .expect("Error in finding edge between two nodes");
        let edge_weight = self.graph.edge_weight(edge_index).unwrap(); //distance, bearing, roadtype

        let ab_bearing = emmit_a.geodesic_bearing(emmit_b);
        let _a_dist = real_point.geodesic_distance(&emmit_a);
        //TODO: if emmit bearing is nan, final bearing is diff is 0
        let _a_bearing =
            (real_point.geodesic_bearing(emmit_a) - ab_bearing + 180.0) % 360.0 - 180.0; //normalize
        let _b_dist = real_point.geodesic_distance(&emmit_b);
        let _b_bearing =
            (real_point.geodesic_bearing(emmit_b) - ab_bearing + 180.0) % 360.0 - 180.0;
        [
            edge_weight.0, // step distance
            edge_weight.1, // step bearing
            // (edge_weight.2 as i32) as f64, //roadtype as float
            edge_weight.2.to_speed(), // speed for roadtype in m/s
                                      // a_dist,
                                      // a_bearing,
                                      // b_dist,
                                      // b_bearing,
        ]
        .to_vec()
    }

    // returns a vector of sets of candidates
    fn candidate_ids_ts(&self, trace: Vec<LonLatTs>, radius: f64) -> Vec<Vec<Candidate>> {
        trace
            .par_iter()
            .map(|lonlatts| {
                let point = Point::<f64>::from((lonlatts.0, lonlatts.1));
                let set_candidates: HashSet<_> = self
                    .graph
                    .node_indices()
                    .filter(|nodeid| {
                        let (_, lon, lat) = self
                            .get_node_weight(nodeid.index())
                            .expect("Error in getting node weight");
                        let nodepoint = Point::<f64>::from((lon, lat));
                        return nodepoint.geodesic_distance(&point) < radius;
                    })
                    .map(|nodeid| {
                        return Candidate {
                            node_index: nodeid,
                            ts: lonlatts.2,
                        };
                    })
                    .collect();
                return set_candidates.into_iter().collect();
            })
            .collect()
    }

    fn multi_yen(
        &self,
        start: &Candidate,
        endnodes: &HashSet<Candidate>,
        k: usize,
    ) -> Vec<(MyNodeIndex, Vec<MyNodeIndex>)> {
        // only care about nodeindex
        let breakoff = u32::max(
            endnodes
                .iter()
                .map(|cand| cand.ts - start.ts)
                .max()
                .expect("Error in finding max time diff")
                * 2,
            10,
        );
        let start = start.node_index;
        let endnodes = endnodes
            .iter()
            .map(|cand| cand.node_index)
            .collect::<HashSet<_>>();
        let successors = |index_capture: &MyNodeIndex| {
            let edgerefs = self
                .graph
                .edges_directed(*index_capture, petgraph::Direction::Outgoing);
            edgerefs.map(|edge| {
                let EdgeWeight(distance, _bearing, roadtype) = edge.weight();
                let traversal_time: OrderedFloat<f64> = (distance / roadtype.to_speed()).into();
                let neighbor_index = edge.target();
                return (neighbor_index, traversal_time);
            })
        };
        let yen_result = pathfinding::prelude::yen_breakoff(
            &start,
            successors,
            |node| endnodes.contains(node),
            k,
            Some(breakoff.into()),
        );
        yen_result
            .into_iter()
            .map(|(path, _)| {
                (
                    path.last()
                        .expect("any path found by yen should not be empty")
                        .clone(),
                    path,
                )
            })
            .collect()
    }
    // unmut function
    fn k_shortest_paths(
        &self,
        cand_a: &Candidate,
        cand_b: &Candidate,
        k: usize,
    ) -> Option<Vec<Vec<Candidate>>> {
        let node_index_a = &cand_a.node_index;
        let node_index_b = &cand_b.node_index;
        // yen forces a transition, so if nodea == nodeb just return costless empty path
        if node_index_a == node_index_b {
            return Some(vec![vec![*cand_a, *cand_b]]);
        }
        let successors = |index_capture: &MyNodeIndex| {
            let edgerefs = self
                .graph
                .edges_directed(*index_capture, petgraph::Direction::Outgoing);
            edgerefs.map(|edge| {
                let EdgeWeight(distance, _bearing, roadtype) = edge.weight();
                let traversal_time: OrderedFloat<f64> = (distance / roadtype.to_speed()).into();
                let neighbor_index = edge.target();
                return (neighbor_index, traversal_time);
            })
        };
        // yen leaves the realm of candidates, as we split it up into index nodes and timediff for the breakoff.
        let yen_result = pathfinding::prelude::yen_breakoff(
            //start
            node_index_a,
            // successors function
            successors,
            //success function
            |node| node == node_index_b,
            // k
            k,
            Some(std::cmp::max(
                ((cand_b.ts - cand_a.ts) as f64 * 2.0).into(),
                OrderedFloat::from(10.0),
            )), // extra time for noise on the input trace
        );
        let start = cand_a.ts;
        let end = cand_b.ts;

        // if yen_result is empty, return None
        if yen_result.len() == 0 {
            return None;
        }

        // map yen_result to just the ids, not the costs
        Some(
            yen_result
                .into_iter()
                .map(|(path, _)| {
                    let step = (end - start) as f64 / (path.len() - 1) as f64;
                    path.into_iter()
                        .enumerate()
                        .map(|(i, cand)| {
                            return Candidate {
                                node_index: cand,
                                ts: (start as f64 + i as f64 * step) as Timestamp,
                            };
                        })
                        .collect::<Vec<_>>()
                })
                .collect(),
        )
    }

    // unmut function
    // returns None if no path is found
    fn prev_to_curr_cost<F>(
        &self,
        start_cand: &Candidate,
        end_cand: &Candidate,
        hiddenstates: &HashMap<Candidate, State>,
        pathcost_func: F, //|a: &Vec<MyNodeIndex>,b: &HashMap<MyNodeIndex, State>, c: &(MyNodeIndex,Timestamp)| -> State
    ) -> Option<State>
    where
        F: Fn(&Self, &Vec<Candidate>, &HashMap<Candidate, State>, &Candidate) -> State,
    {
        let k = self.yendepth;
        let k_shortest_paths = self.k_shortest_paths(&start_cand, &end_cand, k);
        //TODO: processing of features (save in edgeweight) before putting them in pathcost_func (less computation)
        let best_state = k_shortest_paths.map(|paths| {
            paths
                .iter()
                .map(|path| pathcost_func(self, path, hiddenstates, end_cand))
                .min_by_key(|hiddenstate| hiddenstate.cost)
                .expect("Error in finding best path")
        });
        best_state //can be None
    }

    fn single_path_cost(
        &self,
        path: &Vec<Candidate>,
        hiddenstates: &HashMap<Candidate, State>,
        _end_cand: &Candidate,
    ) -> State {
        let mut prev_bearing: f64 = 0.0f64;
        let prev_state = hiddenstates.get(&path[0]);
        match prev_state {
            Some(prev_state) => {
                prev_bearing = prev_state.bearing;
            }
            // do nothing
            None => {}
        }
        // extract the features of the path
        let features: Vec<f64> = path
            .iter()
            .zip(path.into_iter().skip(1))
            .flat_map(|(canda, candb)| {
                let a = canda.node_index;
                let b = candb.node_index;
                let edge_index = self
                    .graph
                    .find_edge(a, b)
                    .expect("Error in finding edge between two nodes");
                let edge_weight = self
                    .graph
                    .edge_weight(edge_index)
                    .expect("if an edge exists, it should have a weight");
                let distance = edge_weight.0;
                let bearing = edge_weight.1;
                let roadtype = edge_weight.2;
                let mut delta_bearing = 0.0f64;
                if !bearing.is_nan() {
                    delta_bearing = ((bearing - prev_bearing) + 180.0) % 360.0 - 180.0;
                }
                let sign_bearing = delta_bearing.signum();
                delta_bearing = delta_bearing.abs();
                // if bearing is nan, we don't update prev_bearing because it's a stand still operation.
                if !bearing.is_nan() {
                    prev_bearing = bearing;
                }
                return vec![distance / roadtype.to_speed(), delta_bearing, sign_bearing];
                // here go extra features: roadtype as type, distance directly, ...
                //units: seconds, degrees
            })
            .collect::<Vec<f64>>();
        // put in ndarray
        let features = Array2::from_shape_vec((path.len() - 1, 3), features)
            .expect("Error in array2 from features");
        // do matrix multiplication //this will later be a NN
        let cost = features.dot(&self.ml_weights.t()).sum();
        // form hidden state
        return State {
            cost: cost.into(),
            bearing: prev_bearing,
            backp: path[0],
            micropath: path.clone(),
        };
    }
}

#[pyfunction]
// takes a bounding box in lat,lon and returns a bounding box in lon,lat which is 'delta_meters' larger in each direction
// bbox order is maxlat, minlat, maxlon, minlon
fn enlarge_bbox(bbox: (f64, f64, f64, f64), delta_meters: f64) -> (f64, f64, f64, f64) {
    let maxlat = bbox.0;
    let minlat = bbox.1;
    let maxlon = bbox.2;
    let minlon = bbox.3;
    let maxlatlon = Point::<f64>::from((maxlon, maxlat));
    let minlatlon = Point::<f64>::from((minlon, minlat));
    let maxlatlon_enlarged = maxlatlon
        .geodesic_destination(90.0, delta_meters)
        .geodesic_destination(0.0, delta_meters);
    let minlatlon_enlarged = minlatlon
        .geodesic_destination(270.0, delta_meters)
        .geodesic_destination(180.0, delta_meters);
    let maxlat_enlarged = maxlatlon_enlarged.y();
    let minlat_enlarged = minlatlon_enlarged.y();
    let maxlon_enlarged = maxlatlon_enlarged.x();
    let minlon_enlarged = minlatlon_enlarged.x();
    return (
        maxlat_enlarged,
        minlat_enlarged,
        maxlon_enlarged,
        minlon_enlarged,
    );
}

#[pyfunction]
// simple perturb trace
fn uniform_perturb_trace(trace: Vec<(f64, f64)>, radius: f64) -> Vec<(f64, f64)> {
    trace
        .par_iter()
        .map(|trace_point| {
            let uniform_random = Uniform::from(0.0..radius);
            let angle_sampler = Uniform::from(0.0..2.0 * PI);
            let mut rng = rand::thread_rng();
            let trace_point = Point::<f64>::from((trace_point.0, trace_point.1));
            let sampled_radius = uniform_random.sample(&mut rng);
            let sampled_angle = angle_sampler.sample(&mut rng).to_degrees();
            let newcoord = trace_point.geodesic_destination(sampled_angle, sampled_radius);
            // return as tuple of lat,lon
            return (newcoord.x(), newcoord.y());
        })
        .collect()
}
#[pyfunction] // py export of the base function
fn prediction_error(pred_path_lonlat: Vec<(f64, f64)>, real_lonlat: Vec<(f64, f64)>) -> f64 {
    // call base
    let predicted_error_fixedpoint = prediction_error_base(&pred_path_lonlat, &real_lonlat);
    match predicted_error_fixedpoint {
        Ok(fixed) => FromFixed::from_fixed(fixed),
        Err(_) => f64::NAN,
    }
}

fn prediction_error_base(
    pred_path_lonlat: &Vec<(f64, f64)>,
    real_lonlat: &Vec<(f64, f64)>,
) -> Result<I32F32, EmptyPredictionError> {
    // assert len > 1 for both
    if pred_path_lonlat.len() <= 1 || real_lonlat.len() <= 1 {
        return Err(EmptyPredictionError);
    }
    // map each pred to closest real POINT
    let map_dist_to_point: Vec<_> = pred_path_lonlat
        .iter()
        // removes consecutive duplicates, makes the comparison pure geometric
        .coalesce(|x, y| if x == y { Ok(x) } else { Err((x, y)) })
        .map(|pred| {
            let point = Point::<f64>::from(*pred);
            let dists = real_lonlat
                .iter()
                .map(|real| I32F32::from_num(point.geodesic_distance(&Point::<f64>::from(*real))));
            let min_dist: I32F32 = dists.min().unwrap();
            return min_dist;
        })
        .collect_vec();
    // mean is annoying because I32F32 doesn't have from usize
    return Ok(map_dist_to_point.iter().sum::<I32F32>() / I32F32::from_num(map_dist_to_point.len()));
}

#[derive(Debug)]
struct EmptyPredictionError;

// bounding_box is minlon, maxlon, minlat, maxlat
fn graph_from_file(
    filepath: &str,
    roadtype_selection: Vec<&str>,
    bounding_box: Option<(f64, f64, f64, f64)>,
    check_oneway: Option<bool>,
) -> PyGraph {
    let check_oneway = match check_oneway {
        Some(check_oneway) => check_oneway,
        None => true,
    };
    let mut parreader = osmpbfreader::OsmPbfReader::new(std::fs::File::open(filepath).unwrap());
    let mut parreader_2 = osmpbfreader::OsmPbfReader::new(std::fs::File::open(filepath).unwrap());
    let mut pygraph = PyGraph::new();
    let graph = &mut pygraph.graph;
    let osmid_to_nodeindex = &mut pygraph.osmid_to_nodeindex;
    let decimicro_to_float = |decimicro: i32| decimicro as f64 / 10000000.0;
    let temp_nodemap: HashMap<i64, NodeInfo> = parreader
        .par_iter()
        .filter_map(|result| match result {
            Ok(obj) => match obj {
                OsmObj::Node(node) => Some((
                    node.id.0,
                    (
                        node.id.0,
                        decimicro_to_float(node.decimicro_lon),
                        decimicro_to_float(node.decimicro_lat),
                    ),
                )),
                _ => None,
            },
            Err(e) => {
                print!("{}", e);
                None
            }
        }) // filter out nodes that are not in the bounding box
        .filter_map(|node: (i64, (i64, f64, f64))| match bounding_box {
            Some(bounding_box) => {
                let (_, lon, lat) = node.1;
                if lon > bounding_box.0
                    && lon < bounding_box.1
                    && lat > bounding_box.2
                    && lat < bounding_box.3
                {
                    Some(node)
                } else {
                    None
                }
            }
            None => Some(node),
        })
        .collect();
    parreader_2.par_iter().for_each(|result| match result {
        Ok(obj) => match obj {
            OsmObj::Way(way) => {
                let mut roadtype : RoadType = RoadType::default();
                // if roadtype selection has been specified, do some filtering otherwise just being a way is enough
                if !roadtype_selection.is_empty() {
                    if !way.tags.contains_key("highway") {
                        return;
                    }
                    let waytype = way.tags.get("highway").unwrap();
                    if !roadtype_selection.iter().any(|&x| waytype.contains(x)) {
                        return;
                    } else {
                        roadtype = match RoadType::from_str(waytype) {
                            Ok(roadtype) => roadtype,
                            Err(_) => return,
                        };
                    }
                }
                let mut node_vec_pairwise: Vec<_> =
                    way.nodes.iter().zip(way.nodes.iter().skip(1)).collect();
                if !check_oneway || way.tags
                    .get("oneway")
                    .map_or(true, |onewaytag| onewaytag != "yes")
                {
                    // this is the reverse of the way because .skip(1) is called on the first iterator
                    let node_rev = way.nodes.iter().skip(1).zip(way.nodes.iter());
                    node_vec_pairwise.extend(node_rev);
                }
                let node_iter_pairwise = node_vec_pairwise.into_iter().map(|(a, b)| (a.0, b.0));
                // for each pair of nodes in the way
                node_iter_pairwise.for_each(|(osma, osmb)| {
                    // another bbox check as we have filtered out nodes that are not in the bbox
                    if !temp_nodemap.contains_key(&osmb) || !temp_nodemap.contains_key(&osma) {
                        return;
                    }
                    // get the nodeinfo from graph
                    let nodew_a = temp_nodemap
                        .get(&osma)
                        .expect("Way specifies node which was not found in the pbf file after bbox selection");
                    let nodew_b = temp_nodemap
                        .get(&osmb)
                        .expect("Way specifies node which was not found in the pbf file after bbox selection");
                    // nodes belong to edge we are interested in
                    // check if the nodes are already in the graph
                    let nodea_id = match osmid_to_nodeindex.get(&osma) {
                        Some(nodea_id) => *nodea_id,
                        None => {
                            let nodea_id = graph.add_node(*nodew_a);
                            // nodes have edge to themselves by default, distance is 0 obviously, bearing is nan
                            graph.add_edge(nodea_id, nodea_id, EdgeWeight::default());
                            osmid_to_nodeindex.insert(osma, nodea_id);
                            nodea_id
                        }
                    };
                    let nodeb_id = match osmid_to_nodeindex.get(&osmb) {
                        Some(nodeb_id) => *nodeb_id,
                        None => {
                            let nodeb_id = graph.add_node(*nodew_b);
                            graph.add_edge(nodeb_id, nodeb_id, EdgeWeight::default());
                            osmid_to_nodeindex.insert(osmb, nodeb_id);
                            nodeb_id
                        }
                    };

                    let pointa: Point<f64> = Point::<f64>::from((nodew_a.1, nodew_a.2));
                    let pointb: Point<f64> = Point::<f64>::from((nodew_b.1, nodew_b.2));
                    // get the distance
                    let distance = pointa.geodesic_distance(&pointb);
                    // get the bearing
                    let bearing = pointa.geodesic_bearing(pointb);
                    let interpolation_distance = 50.0;
                    let num_segments = (distance / interpolation_distance).ceil() as usize;
                    let mut left_id = nodea_id;
                    let step_length = distance / (num_segments as f64);
                    for i in 1..num_segments {
                        let interpolated_point = pointa.geodesic_destination(bearing, i as f64 * step_length );
                        // osmid is 0 and is the default / inexistent osmid
                        let interpolated_node = graph.add_node((0, interpolated_point.x(), interpolated_point.y()));
                        graph.add_edge(interpolated_node, interpolated_node, EdgeWeight::default());
                        graph.add_edge(left_id, interpolated_node, EdgeWeight(distance, bearing, roadtype));
                        left_id = interpolated_node;
                    }
                    // if num_segments is 1, left_id is still nodea_id and a single edge is added
                    graph.add_edge(left_id, nodeb_id, EdgeWeight(distance, bearing, roadtype));
                });
            }
            _ => {}
        },
        Err(_) => {}
    });
    return pygraph;
}

#[pyfunction]
fn reorder_bad_points(
    trace: Vec<(f64, f64, Timestamp)>,
    windowsize: usize,
) -> Vec<(f64, f64, Timestamp)> {
    let mut point_iter = &trace[0];
    let mut result: Vec<(f64, f64, Timestamp)> = [*point_iter].to_vec();
    for i in 0..(trace.len() - windowsize) {
        let window = trace[i..i + windowsize]
            .iter()
            .filter(|window_point| !result.contains(&window_point));
        let closest_in_window = window.min_by_key(|window_point| {
            let pointa = geo::Point::<f64>::from((window_point.0, window_point.1));
            let pointb = geo::Point::<f64>::from((point_iter.0, point_iter.1));
            (pointa.haversine_distance(&pointb) * 10.0) as u64
        });
        match closest_in_window {
            Some(closest_in_window) => {
                point_iter = closest_in_window;
                result.push(*point_iter);
            }
            None => {}
        }
    }
    return result;
}
// export geo::geodesic_distance to python
#[pyfunction]
fn rs_geodesic_distance(pointa: (f64, f64), pointb: (f64, f64)) -> f64 {
    let pointa = geo::Point::<f64>::from((pointa.0, pointa.1));
    let pointb = geo::Point::<f64>::from((pointb.0, pointb.1));
    return pointa.geodesic_distance(&pointb);
}
// export geo::haversine_distance to python
#[pyfunction]
fn rs_haversine_distance(pointa: (f64, f64), pointb: (f64, f64)) -> f64 {
    let pointa = geo::Point::<f64>::from((pointa.0, pointa.1));
    let pointb = geo::Point::<f64>::from((pointb.0, pointb.1));
    return pointa.haversine_distance(&pointb);
}
// export geo::euclidean_distance to python
#[pyfunction]
fn rs_euclidean_distance(pointa: (f64, f64), pointb: (f64, f64)) -> f64 {
    let pointa = geo::Point::<f64>::from((pointa.0, pointa.1));
    let pointb = geo::Point::<f64>::from((pointb.0, pointb.1));
    return pointa.euclidean_distance(&pointb);
}
// export geo::vincenty_distance to python
#[pyfunction]
fn rs_vincenty_distance(pointa: (f64, f64), pointb: (f64, f64)) -> f64 {
    let pointa = geo::Point::<f64>::from((pointa.0, pointa.1));
    let pointb = geo::Point::<f64>::from((pointb.0, pointb.1));
    return match geo::VincentyDistance::vincenty_distance(&pointa, &pointb) {
        Ok(distance) => distance,
        Err(_) => 0.0,
    };
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_tracers")]
fn tracers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(perturb_traces_wsettings, m)?)?;
    m.add_function(wrap_pyfunction!(perturb_traces, m)?)?;
    m.add_function(wrap_pyfunction!(perturb_traces_wpool, m)?)?;
    m.add_function(wrap_pyfunction!(enlarge_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(uniform_perturb_trace, m)?)?;
    m.add_function(wrap_pyfunction!(split_and_filter_traces, m)?)?;
    m.add_function(wrap_pyfunction!(split_traces_on_time, m)?)?;
    m.add_function(wrap_pyfunction!(get_path_length_lonlat, m)?)?;
    m.add_function(wrap_pyfunction!(reorder_bad_points, m)?)?;
    m.add_function(wrap_pyfunction!(rs_geodesic_distance, m)?)?;
    m.add_function(wrap_pyfunction!(rs_haversine_distance, m)?)?;
    m.add_function(wrap_pyfunction!(rs_euclidean_distance, m)?)?;
    m.add_function(wrap_pyfunction!(rs_vincenty_distance, m)?)?;
    m.add_function(wrap_pyfunction!(prediction_error, m)?)?;
    m.add_class::<PyGraph>()?;
    Ok(())
}

#[test]
fn test_split_traces_on_time() {
    let mut traces = Vec::new();
    let mut rng = rand::thread_rng();

    for _ in 0..10 {
        // Generate 10 traces
        let mut trace: Vec<(f64, f64, Timestamp)> = Vec::new();
        let mut timestamp = 0;

        for _ in 0..10 {
            // Each trace has 10 points
            let x = rand::random::<f64>(); // Generate a random f64 for x
            let y = rand::random::<f64>(); // Generate a random f64 for y

            trace.push((x, y, timestamp));

            // Increase timestamp by a random number between 50 and 70
            timestamp += (rand::Rng::gen::<f64>(&mut rng) * 20.0 + 50.0) as Timestamp;
        }

        traces.push(trace);
    }
    println!("{:?}", split_traces_on_time(traces, 60));
}

#[test]
fn test_algo_map_match() {
    let lonlats = vec![
        (7.4132824, 43.7274446, 0),
        (7.4149668, 43.7284912, 100),
        (7.4164581, 43.7297472, 200),
        (7.4188507, 43.7311969, 300),
        (7.4214149, 43.7318249, 400),
        (7.4230134, 43.7321117, 500),
        (7.4216938, 43.7336234, 600),
        (7.4212968, 43.7361196, 700),
        (7.4257171, 43.7376855, 800),
        (7.4252021, 43.7397164, 900),
        (7.4232495, 43.7397242, 1000),
        (7.4217904, 43.7398792, 1100),
    ];
    let mut my_pygraph = PyGraph::new();
    my_pygraph.load_graph(
        "/home/toon/Documents/PHD/coding/osrm-docker/data-volume/monaco/monaco-latest.osm.pbf",
    );
    my_pygraph.yendepth = 1;
    let ts1 = std::time::Instant::now();
    let mapmatch_result = my_pygraph.map_match_trace(lonlats.clone(), 30.0);
    let ts2 = std::time::Instant::now();
    println!("mapmatch_result: {:?}", mapmatch_result);
    println!("time: {:?}", ts2 - ts1);
}

#[test]
fn test_map_match() {
    let latlons = vec![
        (7.4132824, 43.7274446, 0),
        (7.4149668, 43.7284912, 1000),
        (7.4164581, 43.7297472, 2000),
        (7.4188507, 43.7311969, 3000),
        (7.4214149, 43.7318249, 4000),
        (7.4230134, 43.7321117, 5000),
        (7.4216938, 43.7336234, 6000),
        (7.4212968, 43.7361196, 7000),
        (7.4257171, 43.7376855, 8000),
        (7.4252021, 43.7397164, 9000),
        (7.4232495, 43.7397242, 10000),
        (7.4217904, 43.7398792, 11000),
    ];
    let mut my_pygraph = PyGraph::new();
    my_pygraph.load_graph_all(
        "/home/toon/Documents/PHD/coding/osrm-docker/data-volume/monaco/monaco-latest.osm.pbf",
    );
    // let pert_latlon = uniform_perturb_trace(latlons, 10.0);
    let pert_lonlat = latlons;
    let _ts1 = std::time::Instant::now();
    my_pygraph.yendepth = 1;
    let mapmatch_result1 = my_pygraph.map_match_trace(pert_lonlat.clone(), 30.0);
    let _ts2 = std::time::Instant::now();
    // println!("time for yendepth 1: {:?}", ts2 - ts1);
    // println!("mapmatchresult: {:?}", mapmatch_result1);

    let microres = mapmatch_result1.1;
    // flatmap of the microres
    let lonlats = microres
        .iter()
        .flat_map(|optvec| match optvec {
            Some(vec) => vec
                .iter()
                .map(|cand| {
                    let nodeweight = my_pygraph.graph.node_weight(cand.0.into()).unwrap();
                    return (nodeweight.1, nodeweight.2);
                })
                .collect::<Vec<_>>(),
            None => vec![],
        })
        .collect::<Vec<_>>();
    println!("{:?}", lonlats);
}
#[test]
fn test_candidates() {
    //sleep 5 sec
    std::thread::sleep(std::time::Duration::from_secs(15));
    // load graph
    let mut my_pygraph = PyGraph::new();
    my_pygraph.load_graph(
        "/home/toon/Documents/PHD/coding/osrm-docker/data-volume/monaco/monaco-latest.osm.pbf",
    );
    let latlons = vec![(7.4132824, 43.7274446, 0)];
    let candidates = my_pygraph.candidate_ids_ts(latlons, 10.0);
    // get latitude and longitude of candidates
    let latlons: Vec<(f64, f64)> = candidates
        .iter()
        .flatten()
        .map(|a| {
            let nodeweight = my_pygraph.graph.node_weight(a.node_index).unwrap();
            return (nodeweight.1, nodeweight.2);
        })
        .collect();
    println!("{:?}", latlons);
}

#[test]
fn test_china() {
    let mut chinagraph = PyGraph::new();
    chinagraph.yendepth = 1;
    chinagraph.load_graph_base(
        "/home/toon/Documents/PHD/coding/geolife/osmdata/china-latest.osm.pbf",
        vec![
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "residential",
            "unclassified",
        ],
        Some((115.4236, 117.5146, 39.4425, 41.0600)),
        Some(true),
    );
    println!("graph loaded");
    let testtrace = vec![
        (116.318417, 39.984702, 10000),
        (116.31845, 39.984683, 20000),
        (116.318417, 39.984686, 30000),
        (116.318385, 39.984688, 40000),
        (116.318263, 39.984655, 50000),
        (116.318026, 39.984611, 60000),
        (116.317761, 39.984608, 70000),
        (116.317517, 39.984563, 80000),
        (116.317294, 39.984539, 90000),
        (116.317065, 39.984606, 100000),
    ];
    let mapmatch_result = chinagraph.map_match_trace(testtrace, 30.0);
    println!("{:?}", mapmatch_result);
}

// test the 'as 32' of the roadtype enum
#[test]
fn test_roadtype() {
    let roadtypes = vec![
        RoadType::motorway,
        RoadType::trunk,
        RoadType::primary,
        RoadType::secondary,
        RoadType::residential,
    ];
    // all as i32
    let roadtypes_i32: Vec<i32> = roadtypes.iter().map(|a| *a as i32).collect();
    // print
    println!("{:?}", roadtypes_i32);
}

#[test]
fn test_euclidean_point_to_line() {
    // a geo::point
    let zero_point = Point::<f64>::from((0.0, 0.0));
    // a geo::line
    let zero_line = geo::LineString::from(vec![(-100.0, 10.0), (500.0, 10.0), (0.0, -1.0)]);
    let result = zero_point.euclidean_distance(&zero_line);
    println!("{:?}", result);
}

#[test]
fn test_rangefill() {
    let cand_a = Candidate {
        node_index: 0.into(),
        ts: 0,
    };
    let cand_aa = Candidate {
        node_index: 2.into(),
        ts: 0,
    };
    let cand_b = Candidate {
        node_index: 1.into(),
        ts: 7,
    };

    let path = vec![cand_a, cand_aa, cand_aa, cand_aa, cand_aa, cand_b];

    let start = path[0].ts;
    let end = path[path.len() - 1].ts;
    let step = (end - start) as f64 / (path.len() - 1) as f64;
    let tspath = path
        .into_iter()
        .enumerate()
        .map(|(i, cand)| {
            return Candidate {
                node_index: cand.node_index,
                ts: (start as f64 + i as f64 * step) as Timestamp,
            };
        })
        .collect::<Vec<_>>();
    println!("{:?}", tspath);
}

#[test]
fn test_cartesian_product() {
    let candidates = vec![vec![1, 2, 3], vec![], vec![7, 8, 9]];
    let itercand = candidates.iter().map(|a| a.iter());
    let result: Vec<_> = itercand.multi_cartesian_product().collect();
    println!("{:?}", result);
}

#[test]
fn test_none_collection() {
    let input = vec![Some(1), Some(2), Some(3)];
    let output: Option<Vec<_>> = input.into_iter().collect();
    println!("{:?}", output); // Output: Some([1, 2, 3])

    let input_with_none = vec![Some(1), None, Some(3)];
    let output_with_none: Option<Vec<_>> = input_with_none.into_iter().collect();
    println!("{:?}", output_with_none); // Output: None

    let innn = vec![
        vec![vec![1, 2, 3], vec![1, 2, 3]],
        vec![vec![11, 22, 33], vec![11, 22, 33]],
    ];
    println!("{:?}", innn);
    let res = innn
        .into_iter()
        .map(|a| a.into_iter().flatten().collect_vec())
        .collect_vec();
    println!("{:?}", res);
}

#[test]
fn test_report_to_features() {
    let radius = 19.0;
    let sample_ratio = 1.0;
    let mut my_pygraph = PyGraph::new();
    my_pygraph.load_graph_base(
        "/home/toon/Documents/PHD/coding/romataxi/osmdata/centro-160101.osm.pbf",
        [
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "residential",
            "unclassified",
            "service",
            "motorway_link",
            "trunk_link",
            "primary_link",
            "secondary_link",
            "motorway_junction",
        ]
        .to_vec(),
        None,
        None,
    );
    let node = my_pygraph
        .graph
        .node_indices()
        .find(|&node| {
            let nodew = my_pygraph.graph.node_weight(node).unwrap();
            return (nodew.2 - 41.8898608).abs() < 0.0000001
                && (nodew.1 - 12.4364452).abs() < 0.0000001;
        })
        .unwrap();
    println!("{:?}", node);
    println!("{:?}", my_pygraph.graph.node_weight(node).unwrap());
    let neigh = my_pygraph.graph.neighbors(node);
    let someneighs = neigh.collect_vec();
    // print their weights
    let neighweights = someneighs
        .iter()
        .map(|&neigh| my_pygraph.graph.node_weight(neigh).unwrap())
        .collect::<Vec<_>>();
    println!("{:?}", neighweights);
}

#[test]
fn test_prediction_error() {
    // a list of points
    let some_path = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)];
    let other_path = vec![(0.0, 0.0), (0.0, 0.0)];
    let predicted_error = prediction_error_base(&some_path, &other_path);
    println!("{:?}", predicted_error);

    let reports = vec![
        (7.4132824, 43.7274446),
        (7.4149668, 43.7284912),
        (7.4164581, 43.7297472),
        (7.4188507, 43.7311969),
        (7.4214149, 43.7318249),
        (7.4230134, 43.7321117),
        (7.4216938, 43.7336234),
        (7.4212968, 43.7361196),
        (7.4257171, 43.7376855),
        (7.4252021, 43.7397164),
        (7.4232495, 43.7397242),
        (7.4217904, 43.7398792),
    ];

    let other_path = vec![(7.4132824, 43.7274446), (7.4132824, 43.7274446)];
    let predicted_error = prediction_error_base(&reports, &other_path);
    println!("{:?}", predicted_error);

    let prediction_error = prediction_error_base(&reports, &reports);
    println!("{:?}", prediction_error);
}

#[test]
fn test_timeout() {
    let t0 = Instant::now();
    // sleep 5
    std::thread::sleep(std::time::Duration::from_secs(5));
    let toprint = (Instant::now() - t0).as_secs() as usize;
    println!("{:?}", toprint);
}

#[test]
fn test_get_features() {
    let emitted_trace_lonlats: Vec<LonLatTs> = vec![
        (12.4898985, 41.9073689, 1391356132),
        (12.4852986, 41.8967714, 1391356440),
        (12.4752392, 41.8986698, 1391356634),
    ];
    let real_trace_lonlat = vec![
        vec![
            (12.4898985, 41.9073689),
            (12.4902746, 41.9068391),
            (12.4901369, 41.9070358),
            (12.490295, 41.9067722),
            (12.4903169, 41.9067003),
            (12.4903209, 41.9066153),
            (12.4902877, 41.9065448),
            (12.4902533, 41.9064717),
            (12.4901963, 41.9063833),
            (12.4903326, 41.9062806),
            (12.490571800589402, 41.9061227002513),
            (12.490811, 41.9059648),
            (12.4909293, 41.905895),
            (12.4909835, 41.9058604),
            (12.4911369, 41.9057684),
            (12.4913145, 41.9056904),
            (12.491766451048713, 41.90554170089129),
            (12.4922184, 41.905393),
            (12.492688801156845, 41.905235400965616),
            (12.4931592, 41.9050778),
            (12.4932692, 41.9049636),
            (12.4934416, 41.904818),
            (12.493659, 41.904677),
            (12.4937753, 41.9045979),
            (12.4940699, 41.9044231),
            (12.4940699, 41.9044231),
            (12.4940699, 41.9044231),
            (12.4944419, 41.904236),
            (12.4945138, 41.9041934),
            (12.494850551536706, 41.90390095050122),
            (12.4951873, 41.9036085),
            (12.4953307, 41.9034841),
            (12.4954403, 41.9033889),
            (12.4957243, 41.9031408),
            (12.4958487, 41.9029951),
            (12.4958733, 41.9029437),
            (12.4959159, 41.9027366),
            (12.4958301, 41.9026043),
            (12.4957811, 41.9025212),
            (12.495752, 41.9024742),
            (12.4957367, 41.9024495),
            (12.4957095, 41.9024057),
            (12.4956832, 41.9023669),
            (12.4956577, 41.902322),
            (12.4956187, 41.9022744),
            (12.4955848, 41.9022461),
            (12.4955383, 41.902214),
            (12.4954764, 41.9021726),
            (12.495359, 41.9020976),
            (12.4952304, 41.902029),
            (12.494886698819409, 41.90180885051857),
            (12.494543, 41.9015887),
            (12.494185248692743, 41.90135450056204),
            (12.4938275, 41.9011203),
            (12.493525099067716, 41.90092270040158),
            (12.493525099067716, 41.90092270040158),
            (12.493525099067716, 41.90092270040158),
            (12.4932227, 41.9007251),
            (12.4930978, 41.9006435),
            (12.492798399056175, 41.900441450393856),
            (12.492499, 41.9002394),
            (12.492080498227304, 41.89996790076902),
            (12.491662, 41.8996964),
            (12.491395849286011, 41.89952445031101),
            (12.4911297, 41.8993525),
            (12.4910167, 41.8992764),
            (12.4908193, 41.8991436),
            (12.4904818, 41.8989208),
            (12.4903418, 41.8988284),
            (12.4900768, 41.8986505),
            (12.4900573, 41.8986374),
            (12.4900571, 41.8986373),
            (12.4900552, 41.898636),
            (12.4900549, 41.8986358),
            (12.4900226, 41.8986141),
            (12.4896426, 41.8983565),
            (12.4892876, 41.8981217),
            (12.4892364, 41.8980893),
            (12.48884553022473, 41.897834401341726),
            (12.488454663558084, 41.89757950134173),
            (12.4880638, 41.8973246),
            (12.487703748730318, 41.89709855056889),
            (12.4873437, 41.8968725),
            (12.4872215, 41.8967958),
            (12.4870163, 41.8966625),
            (12.4867885, 41.8965195),
            (12.4866321, 41.8964296),
            (12.4864851, 41.8963509),
            (12.4863698, 41.8962974),
            (12.4863156, 41.8962745),
            (12.4862552, 41.8962543),
            (12.4860335, 41.8962347),
            (12.4856899, 41.8962051),
            (12.4856197, 41.8962018),
            (12.485571, 41.8962059),
            (12.485532, 41.8962282),
            (12.4855083, 41.8962591),
            (12.4853912, 41.8965209),
            (12.4853611, 41.8966013),
            (12.4852986, 41.8967714),
        ],
        vec![
            (12.4851544, 41.8971544),
            (12.4851316, 41.8971851),
            (12.4850995, 41.8972105),
            (12.4850593, 41.8972285),
            (12.4850271, 41.8972325),
            (12.4849846, 41.897227),
            (12.484553598607764, 41.89712346828613),
            (12.484122598607764, 41.897019934952795),
            (12.4836916, 41.8969164),
            (12.4834722, 41.8968774),
            (12.4833516, 41.8968587),
            (12.4830196, 41.8968189),
            (12.4828127, 41.8967979),
            (12.4827437, 41.8967913),
            (12.4825288, 41.8967597),
            (12.4822236, 41.8966845),
            (12.4819263, 41.8966128),
            (12.4818105, 41.8965889),
            (12.4817604, 41.8965757),
            (12.481253598343972, 41.89647096890518),
            (12.480746798343969, 41.89636623557185),
            (12.48024, 41.8962615),
            (12.479761199082427, 41.89613865099946),
            (12.4792824, 41.8960158),
            (12.4791883, 41.8960027),
            (12.4789463, 41.8959906),
            (12.4788494, 41.8959851),
            (12.4786658, 41.8959759),
            (12.4784819, 41.8959573),
            (12.478147699958292, 41.89594930048629),
            (12.4778135, 41.8959413),
            (12.4774222, 41.8959372),
            (12.4772614, 41.8959758),
            (12.4771671, 41.8959973),
            (12.476977, 41.8960407),
            (12.4768027, 41.8960805),
            (12.4764885, 41.8961509),
            (12.4764357, 41.896164),
            (12.4763614, 41.896181),
            (12.4760299, 41.8962493),
            (12.4758502, 41.8963102),
            (12.4757694, 41.8963297),
            (12.4755065, 41.8963671),
            (12.4751846, 41.8964055),
            (12.4746949, 41.8964596),
            (12.4745666, 41.896471),
            (12.4744851, 41.8964807),
            (12.4745686, 41.8967369),
            (12.4746138, 41.8968643),
            (12.4746308, 41.8969344),
            (12.4749916, 41.8969808),
            (12.4750881, 41.8969832),
            (12.47508680001253, 41.89729213335002),
            (12.475085500012531, 41.89760106668334),
            (12.4750842, 41.89791),
            (12.474721449902663, 41.89789280057296),
            (12.4743587, 41.8978756),
            (12.4743676, 41.8977606),
            (12.4744156, 41.8975204),
            (12.4744156, 41.8975204),
            (12.4744156, 41.8975204),
            (12.4744678, 41.89723),
            (12.4744854, 41.8970821),
            (12.4745479, 41.8970685),
            (12.4746067, 41.8970558),
            (12.4746308, 41.8969344),
            (12.4749916, 41.8969808),
            (12.4750881, 41.8969832),
            (12.47508680001253, 41.89729213335002),
            (12.475085500012531, 41.89760106668334),
            (12.4750842, 41.89791),
            (12.4750858, 41.8982581),
            (12.4750996, 41.8983136),
            (12.4751301, 41.8983397),
            (12.4752616, 41.8983515),
            (12.4752521, 41.8984638),
            (12.4752513, 41.8984731),
            (12.4752477, 41.8985321),
            (12.4752392, 41.8986698),
            (12.4751999, 41.8988531),
            (12.4751697, 41.8989937),
            (12.4750292, 41.8993981),
            (12.4748388, 41.8993925),
            (12.474450299983937, 41.89938985065716),
            (12.4740618, 41.8993872),
            (12.4738558, 41.8993858),
            (12.4738478, 41.8993521),
            (12.4738589, 41.8991446),
            (12.4738589, 41.8991446),
            (12.4738869, 41.8989426),
            (12.4738909, 41.8989151),
            (12.473919800111027, 41.89866885000893),
            (12.4739487, 41.8984226),
            (12.474051, 41.898428),
            (12.4746063, 41.8984571),
            (12.4748395, 41.8985197),
            (12.4749884, 41.8985221),
            (12.4751981, 41.8985302),
            (12.4752477, 41.8985321),
            (12.4752392, 41.8986698),
        ],
    ];
    let radius: f64 = 10.0;
    let trytakeamt: Option<usize> = Some(1_000_000);
    let timeout: Option<usize> = Some(10);

    let mut graph = PyGraph::new();
    graph.load_graph_base(
        "/home/toon/Documents/PHD/coding/romataxi/osmdata/centro-160101.osm.pbf",
        vec![
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "residential",
            "unclassified",
            "service",
            "motorway_link",
            "trunk_link",
            "primary_link",
            "secondary_link",
            "motorway_junction",
        ],
        Some((12.367, 12.541, 41.79, 41.946)),
        Some(true),
    );

    let features = graph.reports_sample_features_and_paths(
        emitted_trace_lonlats,
        real_trace_lonlat,
        radius,
        trytakeamt,
        timeout,
    );
    println!("{:?}", features);
}

// define the trait mean
trait Mean<T> {
    fn mean(&self) -> T;
}

// implement the trait for Vec<T> where T can be summed and has a length
impl<T> Mean<T> for Vec<T>
where
    T: std::ops::Div<T, Output = T> + Copy + Default + std::iter::Sum<T> + From<usize>,
{
    fn mean(&self) -> T {
        let sum: T = self.iter().cloned().sum();
        let len = T::from(self.len());
        return sum / len;
    }
}
