import typing
from typing_extensions import Protocol
from typing import Optional, Tuple, List
from typing import List, Optional, Tuple
import typing


# class PyGraph(typing.Protocol):
#     yendepth: int
#     ml_weights: int

#     def __init__(self) -> None: ...

#     def load_graph(self, filepath: str) -> None: ...

#     def load_graph_all(self, filepath: str) -> None: ...

#     def load_graph_base(self, filepath: str, roadtype_selection: List[str],
#                         bounding_box: typing.Optional[Tuple[float, float, float, float]]) -> None: ...

#     def map_match_multiple_traces(
#         self, traces: List[List[Tuple[float, float]]], radius: float, outfileprefix: str) -> None: ...

#     def map_match_trace(self, input_trace: List[Tuple[float, float]],
#                         radius: float) -> Tuple[List[Tuple[float, float, int]], List[Optional[List[Tuple[int, int]]]]]: ...

#     def get_node_weight(
#         self, nodeid: int) -> typing.Optional[Tuple[int, float, float]]: ...

#     def osm_to_nodeid(self, osmid: int) -> Optional[int]: ...

#     def extract_trace_transition_features(
#         self,
#         stacked_groundtruth_ids: List[List[int]],
#         emitted_trace_lonlats: List[Tuple[float, float]],
#     ) -> List[List[float]]: ...

#     def reports_sample_features(
#         self,
#         emitted_trace_lonlats: List[Tuple[float, float, int]],
#         real_trace_ids_t: List[List[Tuple[float, float]]],
#         radius: float,
#         try_take_amount: Optional[int],
#         timeout: Optional[int],
#     ) -> List[Tuple[List[List[float]], float]]: ...

#     def reports_sample_features_and_paths(
#         self,
#         emitted_trace_lonlats: List[Tuple[float, float, int]],
#         real_trace_ids_t: List[List[Tuple[float, float]]],
#         radius: float,
#         try_take_amount: Optional[int],
#         timeout: Optional[int],
#     ) -> List[Tuple[List[Tuple[float, float, int]], List[List[float]], float]]: ...

#     def reports_sample_paths(
#         self,
#         emitted_trace_lonlats: List[Tuple[float, float, int]],
#         real_trace_ids_t: List[List[Tuple[float, float]]],
#         radius: float,
#         try_take_amount: Optional[int],
#         timeout: Optional[int],
#     ) -> List[Tuple[List[Tuple[float, float, int]], float]]: ...


def perturb_traces(settings: Tuple[float, int],
                   traces: List[List[Tuple[float, float, int]]],
                   chunk_size: Optional[int] = None,
                   ) -> List[List[Tuple[float, float, int]]]: ...


def perturb_traces_wsettings(
    settings_traces: List[Tuple[Tuple[float, int],
                          List[Tuple[float, float, int]]]],
    chunk_size: Optional[int] = None,
) -> List[Tuple[float, float, int]]: ...


def perturb_traces_wpool(
    settings: Tuple[float, int],
    traces: List[List[Tuple[float, float, int]]],
    pool_size: Optional[int] = None,) -> List[Tuple[float, float, int]]: ...


def split_and_filter_traces(
    traces: List[List[Tuple[float, float, int]]],
    max_time: int,
    min_length: int,
    distance_threshold: float,
) -> List[List[Tuple[float, float, int]]]: ...


def split_traces_on_time(
    traces: List[List[Tuple[float, float, int]]],
    max_time: int
) -> List[List[Tuple[float, float, int]]]: ...


def get_path_length_lonlat(
    lonlats: List[Tuple[float, float]]) -> float: ...


def enlarge_bbox(
    bbox: Tuple[float, float, float, float],
    delta_meters: float,
) -> Tuple[float, float, float, float]: ...


# def graph_to_candidates(
#     candidates: List[Tuple[float, float]],
#     trace: List[Tuple[float, float]],
#     radius: float,
# ) -> List[List[Tuple[float, float]]]: ...


# def graph_to_candidates_ids(
#     candidates: List[Tuple[float, float, int]],
#     trace: List[Tuple[float, float]],
#     radius: float,
# ) -> List[List[int]]: ...


def uniform_perturb_trace(
    trace: List[Tuple[float, float]],
    radius: float
) -> List[Tuple[float, float]]: ...


def reorder_bad_points(trace: List[Tuple[float, float, int]]
                       ) -> List[Tuple[float, float, int]]: ...


def rs_geodesic_distance(
    pointa: Tuple[float, float], pointb: Tuple[float, float]) -> float: ...


def rs_haversine_distance(
    pointa: Tuple[float, float], pointb: Tuple[float, float]) -> float: ...


def rs_euclidean_distance(
    pointa: Tuple[float, float], pointb: Tuple[float, float]) -> float: ...


def rs_vincenty_distance(
    pointa: Tuple[float, float], pointb: Tuple[float, float]) -> float: ...


# def prediction_error(pred_path_lonlat: List[Tuple[float, float]],
#                      real_lonlat: List[Tuple[float, float]]) -> float: ...


# def pytest_report_to_features(
#     graph: PyGraph,
#     emmited_trace_lonlats: List[Tuple[float, float, int]],
#     real_trace_ids: List[int],
#     radius: float
# ) -> List[Tuple[List[List[float]], float]]: ...
