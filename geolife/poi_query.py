import overpy
import numpy as np
api = overpy.Overpass()

def get_closest_poi(target_lon, target_lat):

    query = f'''
    [out:json];
    (
    node["amenity"](around:100,{target_lat},{target_lon});
    node["shop"](around:100,{target_lat},{target_lon});
    way["amenity"](around:100,{target_lat},{target_lon});
    way["shop"](around:100,{target_lat},{target_lon});
    relation["amenity"](around:100,{target_lat},{target_lon});
    relation["shop"](around:100,{target_lat},{target_lon});
    );
    (._;>;);
    out geom body;
    '''
    response = api.query(query)
    poi_set = []

    #TODO: for loop for relations

    for element in response.nodes:
        if element.tags:
            node_element = {
            'id': element.id,
            'lat': float(element.lat),   
            'lon': float(element.lon), 
            'tags': element.tags 
            } 
            poi_set.append(node_element)


    for element in response.ways:
        if element.tags:  # Only include if there are tags
            
            bounds = element.attributes['bounds']
            lat = bounds['minlat']
            lon = bounds['minlon']

            # Create a new dictionary for the way element
            way_element = {
                'type': 'way',
                'id': element.id,
                'lat': float(lat), 
                'lon': float(lon), 
                'tags': element.tags 
            }

            # Append the way_element to poi_set
            poi_set.append(way_element)

    if poi_set:
        for element in poi_set:
            lat =  element['lat']
            lon =  element['lon']
            element['distance'] = np.sqrt((lat - target_lat) ** 2 + (lon - target_lon) ** 2)

        sorted_elements = sorted(poi_set, key=lambda e: e['distance'])
        # Find the closest POI
        for closest_poi in sorted_elements:
            tags = closest_poi['tags']  # Extract tags once to avoid repetitive access
            if 'amenity' in tags or 'shop' in tags:
                # Return the appropriate category based on available tags
                if 'amenity' in tags:
                    return tags['amenity']
                elif 'shop' in tags:
                    return tags['shop'] + " shop"
    
    return "Not found"