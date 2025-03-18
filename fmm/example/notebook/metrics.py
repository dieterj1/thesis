import numpy as np
from shapely.geometry import LineString
from shapely.wkt import loads
from shapely.geometry import Point, LineString, Polygon
from scipy.spatial import cKDTree
import folium
from typing import Tuple, Optional, List, Dict
import math
from shapely.ops import substring
from shapely.geometry.base import BaseGeometry
from scipy.spatial import cKDTree
from math import radians, sin, cos, sqrt, atan2
from haversine import haversine, Unit

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


def spatial_skewing(geom1, geom2, show_map=False):
    """
    Calculate spatial skewing from geometry 2 to geometry 1 (map-matched to ground truth)
    with accurate distance measurement in meters.
    
    Parameters:
    -----------
    geom1 : str or shapely.geometry
        WKT string or Shapely geometry object representing the ground truth
    geom2 : str or shapely.geometry
        WKT string or Shapely geometry object representing the map-matched trace
    show_map : bool, default=False
        If True, returns a folium map visualization showing all distances
        
    Returns:
    --------
    dict or tuple
        If show_map=False: Dictionary containing the unidirectional 2to1 skewing metrics in meters
        If show_map=True: Tuple containing (metrics_dict, folium_map)
    """
    # Convert inputs to Shapely geometries if they are WKT strings
    if not isinstance(geom1, BaseGeometry):
        try:
            geom1 = loads(geom1)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid first geometry: {e}")
    
    if not isinstance(geom2, BaseGeometry):
        try:
            geom2 = loads(geom2)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid second geometry: {e}")
    
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
                # Try to handle other geometry types
                coords = []
                if hasattr(geom, 'geoms'):  # Multi-geometries
                    for subgeom in geom.geoms:
                        coords.extend(extract_coords(subgeom))
                    return np.array(coords)
                else:
                    return np.array([(x, y) for x, y in geom.coords])
            except:
                # Last resort: use representative points
                return np.array([[p.x, p.y] for p in geom.representative_point()])
    
    # Extract coordinates from both geometries
    coords1 = extract_coords(geom1)
    coords2 = extract_coords(geom2)
    
    if len(coords1) == 0 or len(coords2) == 0:
        raise ValueError("One or both geometries have no coordinates")
    
    # Build KDTree for efficient nearest neighbor lookups
    tree1 = cKDTree(coords1)
    
    # Calculate distances from geom2 to geom1 (map-matched to ground truth)
    distances_2to1 = []
    nearest_points = []  # Store the nearest points for mapping
    nearest_indices = []  # Store indices of nearest points
    
    for i, coord in enumerate(coords2):
        # Find nearest point in geom1
        dist, idx = tree1.query(coord)
        nearest = coords1[idx]
        nearest_points.append((coord, nearest))
        nearest_indices.append(idx)
        
        # Calculate haversine distance in meters using the library
        # Note: haversine expects (lat, lon) pairs, while our coords are (lon, lat)
        dist_meters = haversine((coord[1], coord[0]), (nearest[1], nearest[0]), unit=Unit.METERS)
        distances_2to1.append(dist_meters)
    
    # Convert to numpy array for efficient calculations
    distances_2to1 = np.array(distances_2to1)
    
    # Make sure we have valid data for calculations
    if len(distances_2to1) == 0:
        raise ValueError("No valid distance measurements could be calculated")
    
    # Calculate unidirectional metrics (geom2 to geom1)
    results = {
        "mean_meters": float(np.mean(distances_2to1)),
        "min_meters": float(np.min(distances_2to1)),
        "max_meters": float(np.max(distances_2to1))
    }
    
    # Calculate median safely
    if len(distances_2to1) % 2 == 1:  # Odd number of elements
        results["median_meters"] = float(sorted(distances_2to1)[len(distances_2to1) // 2])
    else:  # Even number of elements
        sorted_vals = sorted(distances_2to1)
        mid1 = sorted_vals[len(distances_2to1) // 2 - 1]
        mid2 = sorted_vals[len(distances_2to1) // 2]
        results["median_meters"] = float((mid1 + mid2) / 2)
    
    if show_map:
        # Create a map centered on the average coordinates
        centroid1 = np.mean(coords1, axis=0)
        centroid2 = np.mean(coords2, axis=0)
        center_lat = (centroid1[1] + centroid2[1]) / 2
        center_lon = (centroid1[0] + centroid2[0]) / 2
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
        
        # Add geom1 (ground truth) in blue
        if isinstance(geom1, Point):
            folium.CircleMarker(
                location=[geom1.y, geom1.x],
                radius=5,
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.6,
                popup="Ground Truth"
            ).add_to(m)
        else:
            folium.GeoJson(
                geom1,
                name="Ground Truth",
                style_function=lambda x: {'color': 'blue', 'weight': 3}
            ).add_to(m)
            
            # Add large markers for each point in the ground truth trace
            gt_point_group = folium.FeatureGroup(name="Ground Truth Points")
            for i, (x, y) in enumerate(coords1):
                # Check if this point is used as a nearest point
                is_nearest = i in nearest_indices
                # Use a different style for points that are nearest neighbors
                radius = 7 if is_nearest else 5
                fill_opacity = 0.9 if is_nearest else 0.7
                color = 'darkblue' if is_nearest else 'blue'
                
                folium.CircleMarker(
                    location=[y, x],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=fill_opacity,
                    popup=f"Ground Truth Point {i}"
                ).add_to(gt_point_group)
            gt_point_group.add_to(m)
        
        # Add geom2 (map-matched) in red
        if isinstance(geom2, Point):
            folium.CircleMarker(
                location=[geom2.y, geom2.x],
                radius=5,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                popup="Map-Matched"
            ).add_to(m)
        else:
            folium.GeoJson(
                geom2,
                name="Map-Matched",
                style_function=lambda x: {'color': 'red', 'weight': 3}
            ).add_to(m)
            
            # Add markers for each point in the map-matched trace
            mm_point_group = folium.FeatureGroup(name="Map-Matched Points")
            for i, (x, y) in enumerate(coords2):
                folium.CircleMarker(
                    location=[y, x],
                    radius=4,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.7,
                    popup=f"Map-Matched Point {i}: Distance to nearest Ground Truth: {distances_2to1[i]:.2f} meters"
                ).add_to(mm_point_group)
            mm_point_group.add_to(m)
        
        # Add lines connecting map-matched points to their nearest ground truth points
        connection_group = folium.FeatureGroup(name="Distance Connections")
        for i, ((x2, y2), (x1, y1)) in enumerate(nearest_points):
            folium.PolyLine(
                locations=[[y2, x2], [y1, x1]],
                color='green',
                weight=2,
                opacity=0.7,
                popup=f"Distance: {distances_2to1[i]:.2f} meters"
            ).add_to(connection_group)
        connection_group.add_to(m)
        
        # Add a legend
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; 
            border:2px solid grey; z-index:9999; font-size:14px;
            background-color: white; padding: 10px; border-radius: 5px;">
            <div style="margin-bottom: 5px;">
                <span style="color: blue; font-weight: bold;">●</span> Ground Truth Points
            </div>
            <div style="margin-bottom: 5px;">
                <span style="color: blue; font-weight: bold;">―</span> Ground Truth Line
            </div>
            <div style="margin-bottom: 5px;">
                <span style="color: red; font-weight: bold;">●</span> Map-Matched Points
            </div>
            <div style="margin-bottom: 5px;">
                <span style="color: red; font-weight: bold;">―</span> Map-Matched Line
            </div>
            <div>
                <span style="color: green; font-weight: bold;">―</span> Distance Measurements
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # Add layer control
        folium.LayerControl().add_to(m)
        
        return results, m
    
    return results


def calculate_trace_areas(
    gt_trace: LineString,
    perturbed_trace: LineString,
    angle_threshold: float = 0.2,
    area_threshold: float = 0,
    create_map: bool = False
) -> Tuple[Optional[folium.Map], List[Dict], float]:
    """
    Identify where two traces diverge and converge, then calculate the area between them.
    Filters out areas that are smaller than a specified threshold.
    
    Args:
        gt_trace (LineString): The ground truth trace
        perturbed_trace (LineString): The perturbed trace
        angle_threshold (float): Threshold for considering segments parallel (in radians)
        create_map (bool): Whether to create and return a folium map visualization
        
    Returns:
        Tuple containing:
            - folium.Map or None: Map with traces, intersection points, and enclosed areas (if create_map=True)
            - list: List of areas between divergent sections
            - float: Total area between all divergent sections
    """
    # Convert LineStrings to coordinate lists (lon, lat)
    line1_coords = list(gt_trace.coords)
    line2_coords = list(perturbed_trace.coords)
    
    # Initialize map if requested
    m = None
    if create_map:
        m = _create_map(gt_trace, perturbed_trace, line1_coords, line2_coords)
    
    # Find significant intersections (where traces diverge or converge)
    significant_intersections = _find_intersections(
        gt_trace, perturbed_trace, line1_coords, line2_coords, angle_threshold, m
    )
    
    # Calculate areas between divergent sections
    areas, total_area = _calculate_areas(
        gt_trace, perturbed_trace, significant_intersections, m
    )
    
    # Add layer control and save if map was created
    if create_map and m is not None:
        folium.LayerControl().add_to(m)
    
    return m, areas, total_area


def _create_map(
    gt_trace: LineString,
    perturbed_trace: LineString,
    line1_coords: List,
    line2_coords: List
) -> folium.Map:
    """
    Create a folium map with the two traces.
    
    Args:
        gt_trace: Ground truth trace LineString
        perturbed_trace: Perturbed trace LineString
        line1_coords: Coordinates of ground truth trace
        line2_coords: Coordinates of perturbed trace
        
    Returns:
        folium.Map: Map with both traces
    """
    # Calculate map bounds and center
    def get_bounds(line):
        return (line.bounds[0], line.bounds[1], line.bounds[2], line.bounds[3])

    line1_bounds = get_bounds(gt_trace)
    line2_bounds = get_bounds(perturbed_trace)

    min_lon = min(line1_bounds[0], line2_bounds[0])
    max_lon = max(line1_bounds[2], line2_bounds[2])
    min_lat = min(line1_bounds[1], line2_bounds[1])
    max_lat = max(line1_bounds[3], line2_bounds[3])

    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    # Create Folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    # Function to reverse coordinates for Folium
    def reverse_coords(coords):
        return [(lat, lon) for (lon, lat) in coords]

    # Add original traces to map
    folium.PolyLine(
        reverse_coords(line1_coords),
        color='blue',
        weight=2.5,
        opacity=1,
        tooltip='Ground Truth'
    ).add_to(m)

    folium.PolyLine(
        reverse_coords(line2_coords),
        color='red',
        weight=2.5,
        opacity=1,
        tooltip='Perturbed Path'
    ).add_to(m)
    
    return m


def _find_intersections(
    gt_trace: LineString,
    perturbed_trace: LineString,
    line1_coords: List,
    line2_coords: List,
    angle_threshold: float,
    m: Optional[folium.Map] = None
) -> List[Dict]:
    """
    Find significant intersection points between the two traces.
    
    Args:
        gt_trace: Ground truth trace LineString
        perturbed_trace: Perturbed trace LineString
        line1_coords: Coordinates of ground truth trace
        line2_coords: Coordinates of perturbed trace
        angle_threshold: Threshold for considering segments parallel (in radians)
        m: Folium map to add markers to (optional)
        
    Returns:
        List of significant intersection points with their metadata
    """
    significant_intersections = []
    
    for i in range(len(line1_coords)-1):
        seg1 = LineString([line1_coords[i], line1_coords[i+1]])
        
        for j in range(len(line2_coords)-1):
            seg2 = LineString([line2_coords[j], line2_coords[j+1]])
            
            if seg1.intersects(seg2):
                intersection = seg1.intersection(seg2)
                
                if isinstance(intersection, Point):
                    ip_coords = (intersection.x, intersection.y)
                    
                    # Calculate segment vectors
                    vec1 = np.array([line1_coords[i+1][0] - line1_coords[i][0], 
                                    line1_coords[i+1][1] - line1_coords[i][1]])
                    vec2 = np.array([line2_coords[j+1][0] - line2_coords[j][0],
                                    line2_coords[j+1][1] - line2_coords[j][1]])
                    
                    # Calculate angle between segments
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    
                    if norm1 > 0 and norm2 > 0:
                        cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
                        # Clip to valid range for arccos
                        cos_angle = max(-1.0, min(1.0, cos_angle))
                        angle = np.arccos(cos_angle)
                        
                        # Check if segments are parallel
                        is_parallel = angle < angle_threshold or angle > np.pi - angle_threshold
                        
                        # Only keep intersections where segments are NOT parallel
                        if not is_parallel:
                            # Calculate distance along each line
                            dist1 = gt_trace.project(Point(ip_coords))
                            dist2 = perturbed_trace.project(Point(ip_coords))
                            
                            significant_intersections.append({
                                'point': Point(ip_coords),
                                'location': (ip_coords[1], ip_coords[0]),  # (lat, lon)
                                'segments': (i, j),
                                'angle': angle,
                                'dist1': dist1,  # Distance along gt_trace
                                'dist2': dist2   # Distance along perturbed_trace
                            })
                            
                            # Add marker to the map if provided
                            if m is not None:
                                folium.CircleMarker(
                                    location=(ip_coords[1], ip_coords[0]),
                                    radius=7,
                                    color='#00ff00',
                                    fill=True,
                                    fill_color='#00ff00',
                                    popup=f"Segments: {i}-{j}<br>Angle: {angle:.2f} rad"
                                ).add_to(m)
    
    # Sort intersections by distance along the first line
    significant_intersections.sort(key=lambda x: x['dist1'])
    return significant_intersections


def _calculate_areas(
    gt_trace: LineString,
    perturbed_trace: LineString,
    significant_intersections: List[Dict],
    m: Optional[folium.Map] = None
) -> Tuple[List[Dict], float]:
    """
    Calculate areas between divergent sections of the traces.
    
    Args:
        gt_trace: Ground truth trace LineString
        perturbed_trace: Perturbed trace LineString
        significant_intersections: List of intersection points
        area_threshold: Minimum area in square meters to consider
        m: Folium map to add polygons to (optional)
        
    Returns:
        Tuple containing:
            - list of areas between divergent sections
            - total area between all divergent sections
    """
    # Function to reverse coordinates for Folium
    def reverse_coords(coords):
        return [(lat, lon) for (lon, lat) in coords]
    
    areas = []
    total_area = 0
    
    # For each pair of consecutive intersections
    for i in range(len(significant_intersections) - 1):
        start_point = significant_intersections[i]
        end_point = significant_intersections[i+1]
        
        # Extract the substrings between these points for both lines
        try:
            # Get the distance along each line for start and end points
            start_dist1 = start_point['dist1']
            end_dist1 = end_point['dist1']
            start_dist2 = start_point['dist2']
            end_dist2 = end_point['dist2']
            
            # Create substrings
            segment1 = substring(gt_trace, start_dist1, end_dist1)
            segment2 = substring(perturbed_trace, start_dist2, end_dist2)
            
            # We need to create a polygon from the two segments
            # To do this, we need to ensure the points are in the correct order
            # Start with segment1 from start to end, then segment2 from end to start
            segment1_coords = list(segment1.coords)
            segment2_coords = list(segment2.coords)
            segment2_coords.reverse()  # Reverse to close the polygon
            
            # Create polygon
            polygon_coords = segment1_coords + segment2_coords
            if len(polygon_coords) >= 3:  # Need at least 3 points for a polygon
                polygon = Polygon(polygon_coords)
                
                # Calculate area in square degrees
                area = polygon.area
                
                # Approximate conversion to square meters for lat/lon coordinates
                avg_lat = (start_point['location'][0] + end_point['location'][0]) / 2
                # Convert area in square degrees to square meters
                # 1 degree of latitude is approximately 111,000 meters
                # 1 degree of longitude varies with latitude
                meters_per_degree_lon = 111000 * math.cos(math.radians(avg_lat))
                area_in_sq_meters = area * 111000 * meters_per_degree_lon
                
                
                areas.append({
                    'start_point': start_point['location'],
                    'end_point': end_point['location'],
                    'area_sq_degrees': area,
                    'area_sq_meters': area_in_sq_meters
                })
                    
                total_area += area_in_sq_meters
                    
                # Add polygon to map if provided
                if m is not None:
                    folium.Polygon(
                        locations=reverse_coords(polygon_coords),
                        color='yellow',
                        weight=1,
                        fill=True,
                        fill_color='yellow',
                        fill_opacity=0.4,
                        popup=f"Area: {area_in_sq_meters:.2f} sq meters"
                    ).add_to(m)
        except Exception as e:
            print(f"Error calculating area between points!!! {e}")
            pass
    
    return areas, total_area
