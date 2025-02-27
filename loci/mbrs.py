# MIXTURE-BASED BEST REGION SEARCH

import geopandas as gpd
import pandas as pd
import math

from rtree import index
import networkx as nx
import numpy as np
from statistics import mean, median
import random
from random import sample
import time
from scipy.stats import entropy
import heapq
import folium

import json
from scipy.spatial import ConvexHull, Delaunay
from shapely import geometry
from shapely.geometry import Point, Polygon, box, mapping
from shapely.ops import cascaded_union, polygonize, unary_union



def get_types(gdf):
    """Extracts the types of points and assigns a random color to each type.
    
    Args:
         gdf: A GeoDataFrame containing the input points.
         
    Returns:
          Set of types and corresponding colors.
    """
    
    types = set()
    
    for kwds in gdf['kwds'].tolist():
        types.add(kwds[0])
    
    colors = {t: "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for t in types}
    return types, colors



def compute_score(init, region_size, params):
    """Computes the score of a distribution.
    
    Args:
         init: A vector containing the values of the type distribution.
         region_size: The number of points that constitute the region.
         params: Configuration parameters.
         
    Returns:
          Computed score and relative entropy.
    """
    
    size = sum(init)
    distr = [x / size for x in init]    
    
    rel_se = entropy(distr) / params['settings']['max_se']
    rel_size = region_size / params['variables']['max_size']['current']

    if params['entropy_mode']['current'] == 'high':
        score = rel_se * (rel_size ** params['variables']['size_weight']['current'])
    else:
        rel_se = 1 - rel_se
        score = rel_se * (rel_size ** params['variables']['size_weight']['current'])
    
    return score, rel_se



def create_graph(gdf, eps):
    """Creates the spatial connectivity graph.
    
    Args:
         gdf: A GeoDataFrame containing the input points.
         eps: The spatial distance threshold for edge creation.
         
    Returns:
          A NetworkX graph and an R-tree index over the points.
    """
    
    # create R-tree index
    rtree = index.Index()

    for idx, row in gdf.iterrows():
        left, bottom, right, top = row['geometry'].x, row['geometry'].y, row['geometry'].x, row['geometry'].y
        rtree.insert(idx, (left, bottom, right, top))
    
    # construct the graph
    G = nx.Graph()

    for idx, row in gdf.iterrows():

        # create vertex
        G.add_nodes_from([(idx, {'cat': [gdf.loc[idx]['kwds'][0]]})])

        # retrieve neighbors and create edges
        neighbors = list()
        left, bottom, right, top = row['geometry'].x - eps, row['geometry'].y - eps, row['geometry'].x + eps, row['geometry'].y + eps
        neighbors = [n for n in rtree.intersection((left, bottom, right, top))]
        a = np.array([gdf.loc[idx]['geometry'].x, gdf.loc[idx]['geometry'].y])
        for n in neighbors:
            if idx < n:
                b = np.array([gdf.loc[n]['geometry'].x, gdf.loc[n]['geometry'].y])
                dist = np.linalg.norm(a - b)
                if dist <= eps:
                    G.add_edge(idx, n)
    
    # check max node degree
    cc = [d for n, d in G.degree()]
    max_degree = sorted(cc)[-1] + 1
    mean_degree = mean(cc)
    median_degree = median(cc)
    print('Max degree: ' + str(max_degree) + ' Mean degree: ' + str(mean_degree) + ' Median degree: ' + str(median_degree))
    
    # check connected components
    print('Max connected component: ' + str([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)][0]))
    
    return G, rtree


# Creates a new GRID-based data frame with identical columns as the original dataset
# CAUTION! Assuming that column 'cat1' contains the categories
def partition_data_in_grid(gdf, cell_size):
    """Partitions a GeoDataFrame of points into a uniform grid of square cells.
    
    Args:
         gdf: A GeoDataFrame containing the input points.
         cell_size: The size of the square cell (same units as the coordinates in the input data).
         
    Returns:
          An R-tree index over the input points; also, a GeoDataFrame representing the centroids of the non-empty cells of the grid.
    """
    
    # Spatial extent of the data   
    min_lon, min_lat, max_lon, max_lat = gdf.geometry.total_bounds
    
    # create R-tree index over this dataset of points to facilitate cell assignment
    prtree = index.Index()
        
    for idx, row in gdf.iterrows():
        left, bottom, right, top = row['geometry'].x, row['geometry'].y, row['geometry'].x, row['geometry'].y
        prtree.insert(idx, (left, bottom, right, top))
        
    # Create a data frame for the virtual grid of square cells and keep the categories of points therein
    df_grid = pd.DataFrame(columns=['lon','lat','kwds'])
    numEmptyCells = 0
    for x0 in np.arange(min_lon - cell_size/2.0, max_lon + cell_size/2.0, cell_size):
        for y0 in np.arange(min_lat - cell_size/2.0, max_lat + cell_size/2.0, cell_size):
            # bounds
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            # Get all original points withing this cell from the rtree
            points = list()
            points = [n for n in prtree.intersection((x0, y0, x1, y1))]
            if points:
                subset = gdf.loc[gdf.index.isin(points)]
                # Keep the centroid of each NON-EMPTY cell in the grid
                cell = {'lon':(x0 + x1)/2, 'lat':(y0 + y1)/2, 'kwds':subset['kwds'].map(lambda x: x[0]).tolist()}
                if not cell['kwds']:
                    numEmptyCells += 1
                    continue
                # Append cell to the new dataframe
                df_grid = df_grid.append(cell, ignore_index=True)
            else:
                numEmptyCells += 1
    
    print('Created grid partitioning with ' + str(df_grid.size) + ' non-empty cells containing ' + str(len(np.concatenate(df_grid['kwds']))) + ' points ; ' + str(numEmptyCells) + ' empty cells omitted.')
    
    # Create a GeoDataFrame with all non-empty cell centroids
    gdf_grid = gpd.GeoDataFrame(df_grid, geometry=gpd.points_from_xy(df_grid['lon'], df_grid['lat']))
    gdf_grid = gdf_grid.drop(['lon', 'lat'], axis=1)
    
    return prtree, gdf_grid



def pick_seeds(gdf, seeds_ratio):
    """Selects seed points to be used by the ExpCircles algorithm.
    
    Args:
         gdf: A GeoDataFrame containing the input points.
         seeds_ratio: Percentage of points to be used as seeds.
         
    Returns:
          Set of seed points.
    """
    
    # Pick a sample from the input dataset
    sample = gdf.sample(int(seeds_ratio * len(gdf)))  
    seeds = dict()
    # Keep sample points as centers for the circular expansion when searching around for regions
    for idx, row in sample.iterrows():
        s = len(seeds) + 1
        seeds[s] = Point(row['geometry'].x, row['geometry'].y)
                
    return seeds



########################### INTERNAL HELPER METHODS  ################################
   

def check_cohesiveness(gdf, p, region, eps):
    """Checks if point p is within distance eps from at least one of the points in the region.
    
    Args:
         gdf: A GeoDataFrame containing the input points.
         p: Location of the point to examine.
         region: A list with the the identifiers of the points currently in the region.
         eps: The distance threshold.
                  
    Returns:
         A Boolean value.
    """

    for idx, row in gdf.loc[gdf.index.isin(region)].iterrows():
        if (p.distance(row['geometry']) < eps):
            return True
    return False



def expand_region_with_neighbors(G, region):
    """Expands a given region with its neighboring nodes according to the graph.
    
    Args:
         G: The spatial connectivity graph over the input points.
         region: The set of points currently in the region.
                  
    Returns:
         The expanded region.
    """

    # Collect POIs neighboring a given region according to the graph
    neighbors = [n for v in region for n in list(G[v]) if n not in region]
    
    region_ext = set(region.copy())
    # update region
    for n in neighbors:
        region_ext.add(n)
    
    return region_ext
     

def get_region_score(G, types, region, params):
    """Computes the score of the given region according to the connectivity graph.
    
    Args:
         G: The spatial connectivity graph over the input points.
         types: The set of distinct point types.
         region: The set of points in the region.
         params: The configuration parameters.       
                  
    Returns:
         The score of the region, its relative entropy, and a vector with the values of POI type distribution .
    """

    # Merge optional sublists of POI types into a single list
    lst_cat = [G.nodes[n]['cat'] for n in region]
    categories = [item for sublist in lst_cat for item in sublist]
    init = [categories.count(t) for t in types]
    score, entr = compute_score(init, len(region), params)
    return score, entr, init


## INTERNAL ROUTINE USED BY ALL SEARCH METHODS
def update_topk_list(topk_regions, region_core, region_border, rel_se, score, init, params, start_time, updates):
    """Checks and updates the list of top-k region with a candidate region.
    
    Args:
         topk_regions: The current list of top-k best regions.
         region_core: The set of core points of the candidate region.
         region_border: The set of border points of the candidate region.
         rel_se: The relative entropy of the candidate region.
         score: The score of the candidate region.
         init: A vector containing the values of the type distribution of points the candidate region.
         params: The configuration parameters.
         start_time: The starting time of the execution.
         updates: A structure to hold update times of new results.
                  
    Returns:
         The updated list of the top-k best regions.
    """

    
    # Insert this candidate region into the maxheap of top-k regions according to its score...
    if (score > topk_regions[-1][0]):
        # ...as long as it does NOT significantly overlap with existing regions 
        to_add = True
        cand = set(region_core.union(region_border))   # candidate region (core + border) to examine for overlaps
        discarded = []
        # check degree of overlap with existing regions
        for i in range(len(topk_regions)):
            cur = set(topk_regions[i][2][0].union(topk_regions[i][2][1]))  # existing region (core + border) in the list 
            if (len(cur)>0) and ((len(cur.intersection(cand)) / len(cur) >= params['settings']['overlap_threshold']) or (len(cur.intersection(cand)) / len(cand) >= params['settings']['overlap_threshold'])):
                if score > topk_regions[i][0]:
                    discarded.append(topk_regions[i])
                else:
                    to_add = False
                    break
        if (to_add) and (len(discarded) > 0):
            topk_regions = [e for e in topk_regions if e not in discarded]
                
        # Push this candidate region into a maxheap according its score                           
        if to_add:
            topk_regions.append([score, rel_se, [region_core.copy(), region_border.copy()], init.copy(), len(cand)])
            topk_regions = sorted(topk_regions, key=lambda topk_regions: topk_regions[0], reverse=True)
            # ... at the expense of the one currently having the lowest score
            if (len(topk_regions) > params['settings']['top_k']):
                topk_regions = topk_regions[:-1]
            updates[time.time() - start_time] = topk_regions[-1][0]   # Statistics
        
    return topk_regions
    

###################### EXTRA METHODS FOR MAP VISUALIZATION ###############################

def show_map(gdf, region, colors):
    """Draws the points belonging to a single region on the map. Each point is rendered with a color based on its type.
    
    Args:
         gdf: A GeoDataFrame containing the input points.
         region: The region to be displayed, i.e., a list of the identifiers of its constituent points.
         colors: A list containing the color corresponding to each type.        
         
    Returns:
          A map displaying the top-k regions.
    """
    
    map_settings = {
        'location': [gdf.iloc[0]['geometry'].y, gdf.iloc[0]['geometry'].x],
        'zoom': 12,
        'tiles': 'Stamen toner',
        'marker_size': 20
    }
    
    region = gdf.loc[gdf.index.isin(region)]
    cat = 'cat1'

    m = folium.Map(location=map_settings['location'], zoom_start=map_settings['zoom'], tiles=map_settings['tiles'])
    for idx, row in region.iterrows():
        folium.Circle(
            location=[row['geometry'].y, row['geometry'].x],
            radius=map_settings['marker_size'],
            popup=row['kwds'][0],
            color=colors[row['kwds'][0]],
            fill=True,
            fill_color=colors[row['kwds'][0]],
            fill_opacity=1
        ).add_to(m)

    return m


def show_map_topk_convex_regions(gdf, colors, topk_regions):
    """Draws the convex hull around the points per region on the map. Each point is rendered with a color based on its type.
    
    Args:
         gdf: A GeoDataFrame containing the input points.
         colors: A list containing the color corresponding to each type.
         topk_regions: The list of top-k regions to be displayed.
         
    Returns:
          A map displaying the top-k regions.
    """
    
    map_settings = {
        'location': [gdf.iloc[0]['geometry'].y, gdf.iloc[0]['geometry'].x],
        'zoom': 12,
        'tiles': 'Stamen toner',
        'marker_size': 10
    }
    
    m = folium.Map(location=map_settings['location'], zoom_start=map_settings['zoom'], tiles=map_settings['tiles'])
    
    coords = []
    feature_group = folium.FeatureGroup(name="points")
    for idx, region in enumerate(topk_regions):
        gdf_region = gdf.loc[gdf.index.isin(region[2][0].union(region[2][1]))]
        rank = idx+1
        score = region[0]
        
        # Collect all points belonging to this region...
        pts = []

        # Draw each point selected in the region
        for idx, row in gdf_region.iterrows():
            pts.append([row['geometry'].x, row['geometry'].y])
            coords.append([row['geometry'].y, row['geometry'].x])
            folium.Circle(
                location=[row['geometry'].y, row['geometry'].x],
                radius=map_settings['marker_size'],
                popup=gdf.loc[idx]['kwds'][0],
                color=colors[gdf.loc[idx]['kwds'][0]],
                fill=True,
                fill_color=colors[gdf.loc[idx]['kwds'][0]],
                fill_opacity=1
            ).add_to(feature_group)
        
        # Calculate the convex hull of the points in the regio
        poly = geometry.Polygon([pts[i] for i in ConvexHull(pts).vertices])
        # convert the convex hull to geojson and draw it on the background according to its score
        style_ = {'fillColor': '#ffffbf', 'fill': True, 'lineColor': '#ffffbf','weight': 3,'fillOpacity': (1-0.5*score)}
        geojson = json.dumps({'type': 'FeatureCollection','features': [{'type': 'Feature','properties': {},'geometry': mapping(poly)}]})
        folium.GeoJson(geojson,style_function=lambda x: style_,tooltip='<b>rank:</b> '+str(rank)+'<br/><b>points:</b> '+str(len(pts))+'<br/><b>score:</b> '+str(score)).add_to(m)
    
    # Fit map to the extent of topk-regions
    m.fit_bounds(coords)

    feature_group.add_to(m)

    return m



def show_map_topk_grid_regions(gdf, prtree, colors, gdf_grid, cell_size, topk_regions):
    """Draws the points per grid-based region on the map. Each point is rendered with a color based on its type.
    
    Args:
         gdf: A GeoDataFrame containing the input points.
         prtree: The R-tree index already constructed over the input points.
         colors: A list containing the color corresponding to each type.
         gdf_grid: The grid partitioning (cell centroids with their POI types) created over the input points.
         cell_size: The size of the square cell in the applied grid partitioning (user-specified distance threshold eps).
         topk_regions: The list of top-k grid-based regions to be displayed.
         
    Returns:
          A map displaying the top-k regions along with the grid cells constituting each region.
    """
    
    map_settings = {
        'location': [gdf.iloc[0]['geometry'].y, gdf.iloc[0]['geometry'].x],
        'zoom': 12,
        'tiles': 'Stamen toner',
        'marker_size': 10
    }
    
    m = folium.Map(location=map_settings['location'], zoom_start=map_settings['zoom'], tiles=map_settings['tiles'])
    
    coords = []
    feature_group = folium.FeatureGroup(name="points")
    for idx, region in enumerate(topk_regions):
        gdf_grid_region = gdf_grid.loc[gdf_grid.index.isin(region[2][0].union(region[2][1]))]
        rank = idx+1
        score = region[0]
        
        # Collect all grid cells belonging to this region...
        cells = []
        for idx, row in gdf_grid_region.iterrows():
            b = box(row['geometry'].x - cell_size/2.0, row['geometry'].y - cell_size/2.0, row['geometry'].x + cell_size/2.0, row['geometry'].y + cell_size/2.0)
            cells.append(b)
            
        # Merge these cells into a polygon
        poly = unary_union(cells)
        min_lon, min_lat, max_lon, max_lat = poly.bounds  
        # Convert polygon to geojson and draw it on map according to its score
        style_ = {'fillColor': '#ffffbf', 'fill': True, 'lineColor': '#ffffbf','weight': 3,'fillOpacity': (1-0.5*score)} 
        geojson = json.dumps({'type': 'FeatureCollection','features': [{'type': 'Feature','properties': {},'geometry': mapping(poly)}]})
        folium.GeoJson(geojson,style_function=lambda x: style_,tooltip='<b>rank:</b> '+str(rank)+'<br/><b>cells:</b> '+str(len(cells))+'<br/><b>score:</b> '+str(score)).add_to(m)
        
        # Filter the original points contained within the bounding box of the region ...
        cand = [n for n in prtree.intersection((min_lon, min_lat, max_lon, max_lat))]
        # ... and refine with the exact polygon of the grid-based region
        pts = []
        for c in cand:
            if (poly.contains(Point(gdf.loc[c]['geometry'].x,gdf.loc[c]['geometry'].y))):
                pts.append(c)
        # Draw each point with a color according to its type
        gdf_region = gdf.loc[gdf.index.isin(pts)]
        for idx, row in gdf_region.iterrows():
            coords.append([row['geometry'].y, row['geometry'].x])
            folium.Circle(
                location=[row['geometry'].y, row['geometry'].x],
                radius=map_settings['marker_size'],
                popup=gdf.loc[idx]['kwds'][0],
                color=colors[gdf.loc[idx]['kwds'][0]],
                fill=True,
                fill_color=colors[gdf.loc[idx]['kwds'][0]],
                fill_opacity=1
            ).add_to(feature_group) 
        
    # Fit map to the extent of topk-regions
    m.fit_bounds(coords)

    feature_group.add_to(m)

    return m


############################# CIRCLE-BASED EXPANSION METHOD ############################

def run_exp_circles(gdf, rtree, G, seeds, params, eps, types, topk_regions, start_time, updates):    
    """Executes the ExpCircles algorithm. Employes a priority queue of seeds and expands search in circles of increasing radii around each seed.
    
    Args:
         gdf: A GeoDataFrame containing the input points.
         rtree: The R-tree index constructed over the input points.
         G: The spatial connectivity graph over the input points.
         seeds: The set of seeds to be used.
         params: The configuration parameters.
         eps: The distance threshold.
         types: The set of distinct point types.
         top_regions: A list to hold the top-k results.
         start_time: The starting time of the execution.
         updates: A structure to hold update times of new results.
         
    Returns:
          The list of top-k regions found within the given time budget.
    """
    
    # Priority queue of seeds to explore
    queue = []
    
    # PHASE #1: INITIALIZE QUEUE with seeds (circle centers)
    neighbors = dict()   # Keeps a list per seed of all its (max_size) neighbors by ascending distance
    
    local_size = 2   # Check the seed and its 1-NN

    for s in seeds:
        
        # Keep all (max_size) neighbors around this seed for retrieval during iterations
        neighbors[s] = list(rtree.nearest((seeds[s].x, seeds[s].y, seeds[s].x, seeds[s].y), params['variables']['max_size']['current'])).copy()    
    
        # Retrieve 2-NN points to the current seed
        region = neighbors[s][0:local_size]
        n1 = Point(gdf.loc[region[local_size-2]]['geometry'].x, gdf.loc[region[local_size-2]]['geometry'].y)
        n2 = Point(gdf.loc[region[local_size-1]]['geometry'].x, gdf.loc[region[local_size-1]]['geometry'].y)
        dist_farthest = seeds[s].distance(n2)

        # Drop this seed if its two closest neighbors are more than eps away 
        if (n1.distance(n2) > eps):
            continue
            
        # SCORE ESTIMATION
        region = expand_region_with_neighbors(G, region) # Candidate region is expanded with border points
        if len(region) > params['variables']['max_size']['current']:
            continue 
        
        # Estimate score by applying EXPANSION with neighbors
        score, rel_se, init = get_region_score(G, types, region, params)   

        # update top-k list with this candidate
        topk_regions = update_topk_list(topk_regions, region, set(), rel_se, score, init, params, start_time, updates)
    
        # Push this seed into a priority queue
        heapq.heappush(queue, (-score, (s, local_size, dist_farthest)))
    
    
    # PHASE #2: Start searching for the top-k best regions
    while (time.time() - start_time) < params['variables']['time_budget']['current'] and len(queue) > 0:
        
        # Examine the seed currently at the head of the priority queue
        t = heapq.heappop(queue)
        score, s, local_size, dist_last = -t[0], t[1][0], t[1][1], t[1][2]    
        
        # number of neighbos to examine next
        local_size += 1 

        # check max size
        if local_size > params['variables']['max_size']['current'] or local_size > len(neighbors[s]):
            continue

        # get one more point from its neighbors to construct the new region
        region = neighbors[s][0:local_size]
        p = Point(gdf.loc[region[local_size-1]]['geometry'].x, gdf.loc[region[local_size-1]]['geometry'].y)
        # its distance for the seed
        dist_farthest = seeds[s].distance(p)
   
        # COHESIVENESS CONSTRAINT: if next point is > eps away from all points in the current region of this seed, 
        # skip this point, but keep the seed in the priority queue for further search
        if not check_cohesiveness(gdf, p, neighbors[s][0:local_size-1], eps):   
            del neighbors[s][local_size-1]   # Remove point from neighbors
            heapq.heappush(queue, (-score, (s, local_size-1, dist_last)))
            continue
        
        # RADIUS CONSTRAINT: if next point is > eps away from the most extreme point in the current region, 
        # discard this seed, as no better result can possibly come out of it    
        if (dist_farthest - dist_last > eps):
            continue
            
        # COMPLETENESS CONSTRAINT: Skip this seed if expanded region exceeds max_size
        region = expand_region_with_neighbors(G, region)
        if len(region) > params['variables']['max_size']['current']:
            continue 
    
        # SCORE ESTIMATION by applying EXPANSION with neighbors
        score, rel_se, init = get_region_score(G, types, region, params) 
    
        # update top-k score and region
        topk_regions = update_topk_list(topk_regions, region, set(), rel_se, score, init, params, start_time, updates)
        
        # Push this seed back to the queue
        heapq.heappush(queue, (-score, (s, local_size, dist_farthest)))
  
    # Return top-k regions found within time budget
    return topk_regions

    


############################## GRAPH-EXPANSION METHODS ##################################


def init_queue(G, seeds, types, params, topk_regions, start_time, updates):
    """Initializes the priority queue used for exploration.
    
    Args:
         G: The spatial connectivity graph over the input points.
         seeds: The set of seeds to be used.
         types: The set of distinct point types.
         params: The configuration parameters.
         top_regions: A list to hold the top-k results.
         start_time: The starting time of the execution.
         updates: A structure to hold update times of new results.
                  
    Returns:
         A priority queue to drive the expansion process.
    """

    queue = []

    for v in seeds:
    
        # create region
        region_core = {v}
        region_border = set(G[v])
        region = region_core.union(region_border)
    
        # check if border node is actually core
        border_to_core = set()
        for n in region_border:
            has_new_neighbors = False
            for nn in set(G[n]):
                if nn not in region:
                    has_new_neighbors = True
                    break
            if not has_new_neighbors:
                border_to_core.add(n)
        for n in border_to_core:
            region_border.remove(n)
            region_core.add(n)
        
        # check max size
        if len(region) > params['variables']['max_size']['current']:
            continue
    
        # compute 'init' and score        
        score, rel_se, init = get_region_score(G, types, region, params)
    
        # update top-k regions
        topk_regions = update_topk_list(topk_regions, region_core, region_border, rel_se, score, init, params, start_time, updates)

        # add to queue if border is not empty
        if len(region_border) > 0:
            heapq.heappush(queue, (-score, (region_core.copy(), region_border.copy())))
    
    return queue, topk_regions


def expand_region(G, region_core, region_border, nodes_to_expand, params, types):
    """Expands a given region by adding the given set of nodes.
    
    Args:
         G: The spatial connectivity graph over the input points.
         region_core: The set of core points of the region.
         region_border: The set of border points of the region.
         nodes_to_expand: The set of points to be added.
         params: The configuration parameters.
         types: The set of distinct point types.
                  
    Returns:
         The expanded region and its score.
    """
    
    new_region_core = region_core.copy()
    new_region_border = region_border.copy()
    
    for n in nodes_to_expand:

        # move selected border node to core
        new_region_border.remove(n)
        new_region_core.add(n)

        # find new neighbors and add them to border
        new_neighbors = set(G[n])
        for nn in new_neighbors:
            if nn not in new_region_core:
                new_region_border.add(nn)

    # get the newly formed region
    new_region = new_region_core.union(new_region_border)

    # check if border node is actually core
    border_to_core = set()
    for n in new_region_border:
        has_extra_neighbors = False
        for nn in set(G[n]):
            if nn not in new_region:
                has_extra_neighbors = True
                break
        if not has_extra_neighbors:
            border_to_core.add(n)
    for n in border_to_core:
        new_region_border.remove(n)
        new_region_core.add(n)

    # compute 'init' and score
    score, rel_se, init = get_region_score(G, types, new_region, params)
    
    return new_region, new_region_core, new_region_border, init, score, rel_se



def process_queue(G, queue, topk_regions, params, types, start_time, updates):
    """Selects and expands the next region in the queue.
    
    Args:
         G: The spatial connectivity graph over the input points.
         queue: A priority queue of candidate regions.
         top_regions: A list to hold the top-k results.
         params: The configuration parameters.
         types: The set of distinct point types.
         start_time: The starting time of the execution.
         updates: A structure to hold update times of new results.
                  
    Returns:
         The new state after the expansion.
    """
    
    # POP THE NEXT REGION TO EXPAND
    t = heapq.heappop(queue)
    score, region_core, region_border = -t[0], t[1][0], t[1][1]

    if params['methods']['current'] == 'ExpSingle':   # FIND THE BEST BORDER NODE TO EXPAND

        best_region_core = set()
        best_region_border = set()
        best_region_score = -1
        best_region_rel_se = -1

        for n in region_border:
            
            # expand region with this border point
            new_region, new_region_core, new_region_border, init, new_score, new_rel_se = expand_region(
                G, region_core, region_border, [n], params, types
            )

            # check max size
            if len(new_region) > params['variables']['max_size']['current']:
                continue

            # update top-k regions
            topk_regions = update_topk_list(topk_regions, new_region_core, new_region_border, new_rel_se, new_score, init, params, start_time, updates)
                
            # update current best score
            if new_score > best_region_score and len(new_region_border) > 0:
                best_region_core = new_region_core.copy()
                best_region_border = new_region_border.copy()
                best_region_score = new_score
                best_region_rel_se = new_rel_se

        # ADD THE BEST FOUND NEW REGION TO QUEUE
        if best_region_score > -1:
            heapq.heappush(queue, (-best_region_score, (best_region_core.copy(), best_region_border.copy())))
        
        return best_region_score, topk_regions

    elif params['methods']['current'] == 'ExpAll':   # EXPAND THE ENTIRE BORDER

        # expand region with all border points
        new_region, new_region_core, new_region_border, init, new_score, new_rel_se = expand_region(
            G, region_core, region_border, region_border, params, types
        )
        
        # check max size
        if len(new_region) > params['variables']['max_size']['current']:
            return -1, topk_regions

        # update top-k regions
        topk_regions = update_topk_list(topk_regions, new_region_core, new_region_border, new_rel_se, new_score, init, params, start_time, updates)

        # ADD THE NEW REGION TO QUEUE
        if len(new_region_border) > 0:
            heapq.heappush(queue, (-new_score, (new_region_core.copy(), new_region_border.copy())))
        
        return new_score, topk_regions


def run_exp_hybrid(G, seeds, params, types, topk_regions, start_time, updates):
    """Executes the ExpHybrid algorithm.
    
    Args:
         G: The spatial connectivity graph over the input points.
         seeds: The set of seeds to be used.
         params: The configuration parameters.
         types: The set of distinct point types.
         top_regions: A list to hold the top-k results.
         start_time: The starting time of the execution.
         updates: A structure to hold update times of new results.
                  
    Returns:
         The list of top-k regions found within the given time budget.
    """
    
    # create priority queue for regions
    queue = []
    
    # PART I: For each seed, perform ExpAll
    for v in seeds:
        
        # initialize best local region
        best_region_score = 0
        best_region = set()
    
        # initialize region
        region_core = {v}
        region_border = set(G[v])
        region = region_core.union(region_border)
        
        # expand region until max size
        while len(region) <= params['variables']['max_size']['current'] and len(region_border) > 0 and (time.time() - start_time) < params['variables']['time_budget']['current']:
            
            # compute 'init' and score
            score, rel_se, init = get_region_score(G, types, region, params)
            
            # update top-k regions
            topk_regions = update_topk_list(topk_regions, region_core, region_border, rel_se, score, init, params, start_time, updates)

            # check if local best
            if score > best_region_score:
                best_region_score = score
                best_region = region.copy()
            
            # check if border node is actually core
            border_to_core = set()
            for n in region_border:
                has_new_neighbors = False
                for nn in set(G[n]):
                    if nn not in region:
                        has_new_neighbors = True
                        break
                if not has_new_neighbors:
                    border_to_core.add(n)
            for n in border_to_core:
                region_border.remove(n)
                region_core.add(n)
            
            # expand region with all border points
            region, region_core, region_border, init, score, rel_se = expand_region(
                G, region_core, region_border, region_border, params, types
            )            
        
        # add best found region to queue
        if len(best_region) > 0:
            heapq.heappush(queue, (-best_region_score, best_region))
        
    # PART II: For each seed region, perform ExpSingle
    while len(queue) > 0 and (time.time() - start_time) < params['variables']['time_budget']['current']:
        
        # get the next seed region
        t = heapq.heappop(queue)
        score, seed_region = -t[0], t[1]
        
        # pick a seed
        v = seed_region.pop()
        
        # initialize region
        region_core = {v}
        region_border = set(G[v])
        region = region_core.union(region_border)
        
        # initialize best local region
        best_local_region_score = 0
        
        # expand region until max size
        while len(region) <= params['variables']['max_size']['current'] and len(region_border) > 0 and (time.time() - start_time) < params['variables']['time_budget']['current']:
        
            # compute 'init' and score
            score, rel_se, init = get_region_score(G, types, region, params)
            
            # update top-k regions
            topk_regions = update_topk_list(topk_regions, region_core, region_border, rel_se, score, init, params, start_time, updates)

            # check if border node is actually core
            border_to_core = set()
            for n in region_border:
                has_new_neighbors = False
                for nn in set(G[n]):
                    if nn not in region:
                        has_new_neighbors = True
                        break
                if not has_new_neighbors:
                    border_to_core.add(n)
            for n in border_to_core:
                region_border.remove(n)
                region_core.add(n)

            # find best border point to expand
        
            best_region_core = set()
            best_region_border = set()
            best_region_score = -1
            best_region_rel_se = -1

            for n in region_border:

                # expand region with this border point
                new_region, new_region_core, new_region_border, init, new_score, new_rel_se = expand_region(
                    G, region_core, region_border, [n], params, types
                )

                # check max size
                if len(new_region) > params['variables']['max_size']['current']:
                    continue

                # update top-k regions
                topk_regions = update_topk_list(topk_regions, new_region_core, new_region_border, new_rel_se, new_score, init, params, start_time, updates)

                # update current best score
                if new_score > best_region_score and len(new_region_border) > 0:
                    best_region_core = new_region_core.copy()
                    best_region_border = new_region_border.copy()
                    best_region_score = new_score
                    best_region_rel_se = new_rel_se

            # set current region to best
            region_core = best_region_core
            region_border = best_region_border
            region = region_core.union(region_border)
            
            # update best local score
            if best_region_score > best_local_region_score:
                best_local_region_score = best_region_score
                
        # push back to queue with new score
        if len(seed_region) > 0:
            heapq.heappush(queue, (-best_local_region_score, seed_region))
            
    return topk_regions




def run(gdf, G, rtree, types, params, eps):
    """Computes the top-k high/low mixture regions.
    
    Args:
         gdf: A GeoDataFrame containing the input points.
         G: The spatial connectivity graph over the input points.
         rtree: The R-tree index constructed over the input points.
         types: The set of distinct point types.
         params: The configuration parameters.
         eps: The distance threshold.
                  
    Returns:
         The list of top-k regions detected within the given time budget.
    """
    
#    print('entropy_mode: ' + params['entropy_mode']['current'] + '  method: ' + params['methods']['current'])
    
    # Pick seeds from input points
    if (params['methods']['current'] == 'ExpCircles'):
        seeds = pick_seeds(gdf, params['settings']['seeds_ratio'])
    else:
        seeds = sample(list(G.nodes), int(params['settings']['seeds_ratio'] * len(list(G.nodes))))
            
    start_time = time.time()
    
    # Initialize top-k list
    topk_regions = []
    while len(topk_regions) < params['settings']['top_k']:
        topk_regions.append([0, 0, [set(), set()], [], 0])   # [score, rel_se, [region_core, region_border], init, length]

    iterations = 0
    updates = dict()
    
    if params['methods']['current'] == 'ExpHybrid':
        topk_regions = run_exp_hybrid(G, seeds, params, types, topk_regions, start_time, updates)
    elif params['methods']['current'] == 'ExpCircles':
        topk_regions = run_exp_circles(gdf, rtree, G, seeds, params, eps, types, topk_regions, start_time, updates)
    else:   # ExpSingle or ExpAll methods
        queue, topk_regions = init_queue(G, seeds, types, params, topk_regions, start_time, updates)
        # Process queue
        while (time.time() - start_time) < params['variables']['time_budget']['current'] and len(queue) > 0:
            iterations += 1
            score, topk_regions = process_queue(G, queue, topk_regions, params, types, start_time, updates)
    
#    print('Execution time: ' + str(time.time() - start_time)+'sec')
    
    return topk_regions, updates
