import numpy as np
from shapely.geometry import LineString
from shapely.wkt import loads
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.wkt import loads
from scipy.spatial import cKDTree

def precision_length(matched_geometry, ground_truth_geometry):
    """
    Calculate precision by length: the ratio of correctly matched road length to total matched length
    
    Parameters:
    -----------
    matched_geometry : str or LineString
        WKT string or Shapely LineString of the matched path
    ground_truth_geometry : str or LineString
        WKT string or Shapely LineString of the ground truth path
    
    Returns:
    --------
    float: Precision score (0-1)
    """
    # Convert WKT strings to Shapely geometries if needed
    if isinstance(matched_geometry, str):
        matched_geometry = loads(matched_geometry)
    if isinstance(ground_truth_geometry, str):
        ground_truth_geometry = loads(ground_truth_geometry)
    
    # Calculate intersection length
    intersection_length = ground_truth_geometry.intersection(matched_geometry).length
    
    # Calculate total matched length
    matched_length = matched_geometry.length
    
    # Calculate precision (avoid division by zero)
    precision = intersection_length / matched_length if matched_length > 0 else 0
    
    return precision

def recall_length(matched_geometry, ground_truth_geometry):
    """
    Calculate recall by length: the ratio of correctly matched road length to total ground truth length
    
    Parameters:
    -----------
    matched_geometry : str or LineString
        WKT string or Shapely LineString of the matched path
    ground_truth_geometry : str or LineString
        WKT string or Shapely LineString of the ground truth path
    
    Returns:
    --------
    float: Recall score (0-1)
    """
    # Convert WKT strings to Shapely geometries if needed
    if isinstance(matched_geometry, str):
        matched_geometry = loads(matched_geometry)
    if isinstance(ground_truth_geometry, str):
        ground_truth_geometry = loads(ground_truth_geometry)
    
    # Calculate intersection length
    intersection_length = ground_truth_geometry.intersection(matched_geometry).length
    
    # Calculate total ground truth length
    ground_truth_length = ground_truth_geometry.length
    
    # Calculate recall (avoid division by zero)
    recall = intersection_length / ground_truth_length if ground_truth_length > 0 else 0
    
    return recall

def precision_segment(matched_segments, ground_truth_segments):
    """
    Calculate precision by segment: the ratio of correctly matched segments to total matched segments
    
    Parameters:
    -----------
    matched_segments : list or set
        Set of segment IDs in the matched path
    ground_truth_segments : list or set
        Set of segment IDs in the ground truth path
    
    Returns:
    --------
    float: Precision score (0-1)
    """
    # Convert to sets if lists are provided
    if isinstance(matched_segments, list):
        matched_segments = set(matched_segments)
    if isinstance(ground_truth_segments, list):
        ground_truth_segments = set(ground_truth_segments)
    
    # Calculate correctly matched segments
    correctly_matched = len(matched_segments.intersection(ground_truth_segments))
    
    # Calculate total matched segments
    total_matched = len(matched_segments)
    
    # Calculate precision (avoid division by zero)
    precision = correctly_matched / total_matched if total_matched > 0 else 0
    
    return precision

def recall_segment(matched_segments, ground_truth_segments):
    """
    Calculate recall by segment: the ratio of correctly matched segments to total ground truth segments
    
    Parameters:
    -----------
    matched_segments : list or set
        Set of segment IDs in the matched path
    ground_truth_segments : list or set
        Set of segment IDs in the ground truth path
    
    Returns:
    --------
    float: Recall score (0-1)
    """
    # Convert to sets if lists are provided
    if isinstance(matched_segments, list):
        matched_segments = set(matched_segments)
    if isinstance(ground_truth_segments, list):
        ground_truth_segments = set(ground_truth_segments)
    
    # Calculate correctly matched segments
    correctly_matched = len(matched_segments.intersection(ground_truth_segments))
    
    # Calculate total ground truth segments
    total_ground_truth = len(ground_truth_segments)
    
    # Calculate recall (avoid division by zero)
    recall = correctly_matched / total_ground_truth if total_ground_truth > 0 else 0
    
    return recall

def route_error(matched_geometry, ground_truth_geometry):
    """
    Calculate route error based on the paper "Map-matching for low-sampling-rate GPS trajectories"
    by Lou et al. (https://doi.org/10.1145/1653771.1653818)
    
    Parameters:
    -----------
    matched_geometry : str or LineString
        WKT string or Shapely LineString of the matched path
    ground_truth_geometry : str or LineString
        WKT string or Shapely LineString of the ground truth path
    
    Returns:
    --------
    dict: Dictionary containing:
        - error: The final error metric (d_minus + d_plus) / d0
        - d0: Correct route length
        - d_minus: Erroneously subtracted length
        - d_plus: Erroneously added length
    """
    # Convert WKT strings to Shapely geometries if needed
    if isinstance(matched_geometry, str):
        matched_geometry = loads(matched_geometry)
    if isinstance(ground_truth_geometry, str):
        ground_truth_geometry = loads(ground_truth_geometry)
    
    # Compute correct route length (d0)
    d0 = ground_truth_geometry.length
    
    # Compute erroneously subtracted length (d-)
    d_minus_geom = ground_truth_geometry.difference(matched_geometry)
    d_minus = d_minus_geom.length if not d_minus_geom.is_empty else 0
    
    # Compute erroneously added length (d+)
    d_plus_geom = matched_geometry.difference(ground_truth_geometry)
    d_plus = d_plus_geom.length if not d_plus_geom.is_empty else 0
    
    # Compute the final error metric
    error = (d_minus + d_plus) / d0 if d0 > 0 else float('inf')
    
    return {
        "error": error,
        "d0": d0,
        "d_minus": d_minus,
        "d_plus": d_plus
    }


import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.wkt import loads
from scipy.spatial import cKDTree

def compute_spatial_skewing_in_meters(original_geom, pert_matched_geom, avg_latitude=None):
    """
    Compute the spatial skewing between original geometry and perturbed matched geometry,
    with results in meters for coordinates in decimal degrees.
    
    Parameters:
    -----------
    original_geom : Shapely geometry
        The original/ground truth geometry
    pert_matched_geom : Shapely geometry
        The perturbed matched geometry
    avg_latitude : float, optional
        Average latitude of the data for more accurate conversion to meters
        If None, will calculate from the data
    
    Returns:
    --------
    dict
        Dictionary containing mean, median, and maximum skewing distances in meters
    """
    # Extract coordinates based on geometry type
    def extract_coords(geom):
        if isinstance(geom, Point):
            return np.array([[geom.x, geom.y]])
        elif isinstance(geom, LineString):
            return np.array(geom.coords)
        elif isinstance(geom, Polygon):
            return np.array(geom.exterior.coords)
        else:
            try:
                # Try to extract coordinates from any geometry
                return np.array([(x, y) for x, y in geom.coords])
            except:
                raise ValueError(f"Unsupported geometry type: {type(geom)}")
    
    original_coords = extract_coords(original_geom)
    pert_coords = extract_coords(pert_matched_geom)
    
    # Calculate average latitude if not provided
    if avg_latitude is None and len(original_coords) > 0:
        avg_latitude = np.mean(original_coords[:, 1])
    
    # Build KDTree for fast nearest-neighbor lookup
    tree = cKDTree(pert_coords)
    
    # Query the nearest neighbor distances for all original points
    distances_deg, _ = tree.query(original_coords)
    
    # Convert degrees to meters (varies with latitude)
    # At equator: 1 degree = 111,000 meters (roughly)
    meters_per_degree_lat = 111000  # Approximate for latitude
    
    # Longitude conversion depends on latitude
    if avg_latitude is not None:
        meters_per_degree_lon = 111000 * np.cos(np.radians(avg_latitude))
    else:
        meters_per_degree_lon = 111000  # Default to equator
    
    # Since our distances are Euclidean in degree space, use an average conversion factor
    # For more precision, would need to calculate separate lat/lon components
    meters_per_degree_avg = (meters_per_degree_lat + meters_per_degree_lon) / 2
    
    # Convert to meters
    distances_meters = distances_deg * meters_per_degree_avg
    
    # Compute spatial skewing statistics
    mean_skewing = np.mean(distances_meters)
    median_skewing = np.median(distances_meters)
    max_skewing = np.max(distances_meters)
    min_skewing = np.min(distances_meters)
    
    return {
        "mean_meters": mean_skewing,
        "median_meters": median_skewing,
        "max_meters": max_skewing,
        "min_meters": min_skewing
    }

def compute_bidirectional_spatial_skewing_meters(original_geom, pert_matched_geom, avg_latitude=None):
    """
    Compute bidirectional spatial skewing between original and perturbed geometries,
    with results in meters for coordinates in decimal degrees.
    
    Parameters:
    -----------
    original_geom : Shapely geometry
        The original/ground truth geometry
    pert_matched_geom : Shapely geometry
        The perturbed matched geometry
    avg_latitude : float, optional
        Average latitude for more accurate conversion to meters
    
    Returns:
    --------
    dict
        Dictionary containing mean and maximum bidirectional skewing in meters
    """
    # Compute original → perturbed distances
    orig_to_pert = compute_spatial_skewing_in_meters(original_geom, pert_matched_geom, avg_latitude)
    
    # Compute perturbed → original distances
    pert_to_orig = compute_spatial_skewing_in_meters(pert_matched_geom, original_geom, avg_latitude)
    
    # Bidirectional mean: average of both directions
    mean_bidirectional = (orig_to_pert["mean_meters"] + pert_to_orig["mean_meters"]) / 2
    
    # Bidirectional max: max of both directions (similar to Hausdorff distance)
    max_bidirectional = max(orig_to_pert["max_meters"], pert_to_orig["max_meters"])
    
    return {
        "mean_bidirectional_meters": mean_bidirectional,
        "max_bidirectional_meters": max_bidirectional
    }

def analyze_spatial_skewing_meters(wkt, wkt_pert, result_pert=None):
    """
    Analyze spatial skewing between original and perturbed geometries,
    reporting results in meters.
    
    Parameters:
    -----------
    wkt : str
        WKT string of original geometry
    wkt_pert : str
        WKT string of perturbed geometry
    result_pert : object, optional
        Object with mgeom attribute containing matched geometry
        
    Returns:
    --------
    dict
        Dictionary containing both unidirectional and bidirectional skewing metrics
    """
    # Load geometries
    original_geom = loads(wkt)
    pert_geom = loads(wkt_pert)
    
    # Use matched geometry if available, otherwise use perturbed geometry
    if result_pert is not None:
        try:
            pert_matched_geom = loads(result_pert.mgeom.export_wkt())
        except (AttributeError, ValueError):
            print("Warning: Could not load matched geometry, using perturbed geometry instead")
            pert_matched_geom = pert_geom
    else:
        pert_matched_geom = pert_geom
    
    # Calculate average latitude for better meter conversion
    try:
        coords = np.array(original_geom.coords)
        avg_latitude = np.mean(coords[:, 1])
    except:
        avg_latitude = None
    
    # Compute unidirectional skewing
    unidirectional = compute_spatial_skewing_in_meters(original_geom, pert_matched_geom, avg_latitude)
    
    # Compute bidirectional skewing
    bidirectional = compute_bidirectional_spatial_skewing_meters(original_geom, pert_matched_geom, avg_latitude)
    
    # Combine results
    results = {
        "unidirectional": unidirectional,
        "bidirectional": bidirectional
    }
    
    # Print basic results
    print(f"Spatial Skewing")
    print(f"  Mean: {unidirectional['mean_meters']:.2f} meters")
    print(f"  Median: {unidirectional['median_meters']:.2f} meters")
    print(f"  Maximum: {unidirectional['max_meters']:.2f} meters")
    
    return results


def analyze_spatial_skewing_metersV2(wkt, wkt_pert):
    """
    Analyze spatial skewing between original and perturbed geometries,
    reporting results in meters.
    
    Parameters:
    -----------
    wkt : str or shapely.geometry.LineString
        WKT string or LineString geometry of original geometry
    wkt_pert : str or shapely.geometry.LineString
        WKT string or LineString geometry of perturbed geometry
        
    Returns:
    --------
    dict
        Dictionary containing both unidirectional and bidirectional skewing metrics
    """
    # Load geometries, check if they are already geometry objects
    from shapely.geometry.base import BaseGeometry
    
    # Handle original geometry
    if isinstance(wkt, BaseGeometry):
        original_geom = wkt
    else:
        try:
            original_geom = loads(wkt)
        except TypeError as e:
            print(f"Error loading original geometry: {e}")
            raise
    
    # Handle perturbed geometry
    if isinstance(wkt_pert, BaseGeometry):
        pert_geom = wkt_pert
    else:
        try:
            pert_geom = loads(wkt_pert)
        except TypeError as e:
            print(f"Error loading perturbed geometry: {e}")
            raise
    
    # Calculate average latitude for better meter conversion
    try:
        coords = np.array(original_geom.coords)
        avg_latitude = np.mean(coords[:, 1])
    except:
        avg_latitude = None
    
    # Compute unidirectional skewing
    unidirectional = compute_spatial_skewing_in_meters(original_geom, pert_geom, avg_latitude)
    
    # Compute bidirectional skewing
    bidirectional = compute_bidirectional_spatial_skewing_meters(original_geom, pert_geom, avg_latitude)
    
    # Combine results
    results = {
        "unidirectional": unidirectional,
        "bidirectional": bidirectional
    }
    
    # Print basic results
    print(f"Spatial Skewing")
    print(f"  Mean (unidirectional): {unidirectional['mean_meters']:.2f} meters")
    print(f"  Median (unidirectional): {unidirectional['median_meters']:.2f} meters")
    print(f"  Maximum (unidirectional): {unidirectional['max_meters']:.2f} meters")
    print(f"  Maximum (bidirectional): {bidirectional['max_bidirectional_meters']:.2f} meters")
    
    return results


def analyze_spatial_skewing_metersV3(original_geom, perturbed_geom, avg_latitude=None):
    """
    Compute spatial skewing between original and perturbed geometries in meters.
    
    Parameters:
    -----------
    original_geom : LineString
        Original geometry
    perturbed_geom : LineString
        Perturbed geometry
    avg_latitude : float, optional
        Average latitude for better meter conversion
        
    Returns:
    --------
    dict
        Dictionary with skewing metrics
    """
    from shapely.ops import nearest_points
    import numpy as np
    
    # Function to convert degrees to meters
    def haversine_distance(lon1, lat1, lon2, lat2):
        """Calculate haversine distance between two points in meters"""
        from math import radians, sin, cos, sqrt, atan2
        
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        r = 6371000  # Earth radius in meters
        return c * r
    
    # Get distances from original to perturbed
    distances = []
    
    # Extract coordinates
    try:
        orig_coords = list(original_geom.coords)
    except AttributeError:
        # Handle MultiLineString or other geometry types
        print("Warning: Original geometry is not a LineString. Using representative point.")
        orig_coords = [original_geom.representative_point().coords[0]]
    
    # Calculate distances for each point in original geometry
    for point in orig_coords:
        # Create Point object for the original point
        from shapely.geometry import Point
        orig_point = Point(point)
        
        # Find nearest point on perturbed geometry
        nearest = nearest_points(orig_point, perturbed_geom)[1]
        
        # Calculate distance in meters
        if avg_latitude is not None:
            # Use average latitude for more accurate conversion
            dist = haversine_distance(
                point[0], point[1], 
                nearest.x, nearest.y
            )
        else:
            # Fallback to Euclidean distance (less accurate)
            dist = orig_point.distance(nearest) * 111000  # rough conversion to meters
            
        distances.append(dist)
    
    # Ensure we have distances to calculate
    if not distances:
        print("Warning: No distances calculated")
        return {
            "mean_meters": 0,
            "median_meters": 0,
            "max_meters": 0,
            "distances_meters": []
        }
    
    # Calculate metrics
    # Add debugging output
    print(f"DEBUG: Number of distance points: {len(distances)}")
    print(f"DEBUG: First few distances: {distances[:5]}")
    print(f"DEBUG: Distance range: {min(distances)} to {max(distances)}")
    
    # Calculate metrics (with additional checks)
    distances_array = np.array(distances)
    mean_dist = float(np.mean(distances_array))
    median_dist = float(np.median(distances_array))
    max_dist = float(np.max(distances_array))
    
    return {
        "mean_meters": mean_dist,
        "median_meters": median_dist,
        "max_meters": max_dist,
        "distances_meters": distances
    }