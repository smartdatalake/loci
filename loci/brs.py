from rtree import index
import networkx as nx
import numpy as np
from statistics import mean, median
import random
import time
from scipy.stats import entropy
import heapq
import folium


import json
from scipy.spatial import ConvexHull, Delaunay
from shapely import geometry
from shapely.geometry import Point, Polygon, box, mapping
from shapely.ops import cascaded_union, polygonize


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
        G.add_nodes_from([(idx, {'cat': gdf.loc[idx]['kwds'][0]})])

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


def compute_score(init, params):
    """Computes the score of a distribution.
    
    Args:
         init: A vector containing the values of the type distribution.
         params: Configuration parameters.
         
    Returns:
          Computed score and relative entropy.
    """
    
    size = sum(init)
    distr = [x / size for x in init]    
    
    rel_se = entropy(distr) / params['settings']['max_se']
    rel_size = size / params['variables']['max_size']['current']

    if params['entropy_mode']['current'] == 'high':
        score = rel_se * (rel_size ** params['variables']['size_weight']['current'])
    else:
        rel_se = 1 - rel_se
        score = rel_se * (rel_size ** params['variables']['size_weight']['current'])
    
    return score, rel_se



#################### EXTRA METHODS USED IN CIRCULAR SEARCH #######################
   
# Pick a sample from the input dataset as seeds (circle centers to search around for regions)
def pickSeeds(gdf, seeds_ratio):
    """Selects seed points to be used by the ExpCircles algorithm.
    
    Args:
         gdf: A GeoDataFrame containing the input points.
         seeds_ratio: Percentage of points to be used as seeds.
         
    Returns:
          Set of seed points.
    """
    
    sample = gdf.sample(int(seeds_ratio * len(gdf)))  
    seeds = dict()
    for idx, row in sample.iterrows():
        s = len(seeds) + 1
        seeds[s] = Point(row['geometry'].x, row['geometry'].y)
                
    return seeds


# Check if point p is within distance eps from at least one of the current items in the list
def checkCohesiveness(gdf, p, items, eps):
    for idx, row in gdf.loc[gdf.index.isin(items)].iterrows():
        if (p.distance(row['geometry']) < eps):
            return True
    return False


# Helper functions used in score computation

def get_neighbors(G, region):
    return [n for v in region for n in list(G[v]) if n not in region]


def neighbor_extension(G, region):
    # get new neighbors
    neighbors = get_neighbors(G, region)
    
    region_ext = set(region.copy())
    # update region
    for n in neighbors:
        region_ext.add(n)
    
    return region_ext
     

# SCORE from graph
def get_region_score_graph(G, types, region, params):
    categories = [G.nodes[n]['cat'] for n in region]
    init = [categories.count(t) for t in types]
    score, entr = compute_score(init, params)
    return score, entr, init
    
    
    
## MAIN METHOD
## uses a priority queue of seeds and expands search in circles of increasing radii around each seed
def run_exp_circles(gdf, rtree, G, seeds, params, types, topk_regions, start_time, updates):
    """Executes the ExpCircles algorithm.
    
    Args:
         gdf: A GeoDataFrame containing the input points.
         rtree: The R-tree index constructed over the input points.
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
        if (n1.distance(n2) > params['variables']['eps']['current']):
            continue
            
        # SCORE ESTIMATION
        region = neighbor_extension(G, region) # Candidate region is expanded with border points
        if len(region) > params['variables']['max_size']['current']:
            continue 
        score, rel_se, init = get_region_score_graph(G, types, region, params)   # Estimate score by applying EXPANSION with neighbors

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
        if not checkCohesiveness(gdf, p, neighbors[s][0:local_size-1], params['variables']['eps']['current']):   
            del neighbors[s][local_size-1]   # Remove point from neighbors
            heapq.heappush(queue, (-score, (s, local_size-1, dist_last)))
            continue
        
        # RADIUS CONSTRAINT: if next point is > eps away from the most extreme point in the current region, 
        # discard this seed, as no better result can possibly come out of it    
        if (dist_farthest - dist_last > params['variables']['eps']['current']):
            continue
            
        # COMPLETENESS CONSTRAINT: Skip this seed if expanded region exceeds max_size
        region = neighbor_extension(G, region)
        if len(region) > params['variables']['max_size']['current']:
            continue 
    
        # SCORE ESTIMATION by applying EXPANSION with neighbors
        score, rel_se, init = get_region_score_graph(G, types, region, params) 
    
        # update top-k score and region
        topk_regions = update_topk_list(topk_regions, region, set(), rel_se, score, init, params, start_time, updates)
        
        # Push this seed back to the queue
        heapq.heappush(queue, (-score, (s, local_size, dist_farthest)))
  
    # Return top-k regions found within time budget
    return topk_regions



## ROUTINE USED BY ALL SEARCH METHODS
## pushes an entry to the list of top-k regions
## data *must* contain a key 'score', and can contain any other piece of information
def update_topk_list(topk_regions, region_core, region_border, rel_se, score, init, params, start_time, updates):
    
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
#            if ((len(topk_regions) < params['settings']['top_k']) or (score > topk_regions[params['settings']['top_k'] - 1][0])):
            topk_regions.append([score, rel_se, [region_core.copy(), region_border.copy()], init.copy(), len(cand)])
            topk_regions = sorted(topk_regions, key=lambda topk_regions: topk_regions[0], reverse=True)
            # ... at the expense of the one currently having the lowest score
            if (len(topk_regions) > params['settings']['top_k']):
                topk_regions = topk_regions[:-1]
            updates[time.time() - start_time] = topk_regions[-1][0]   # Statistics
        
    return topk_regions
        

###################### EXTRA METHOD FOR VISUALIZATION ###############################

# Draw CONVEX HULL around points per region on map; each point rendered with a color based on its category
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
        style_ = {'fillColor': '#f5f5f5', 'fill': True, 'lineColor': '#ffffbf','weight': 3,'fillOpacity': (1- score)}
        geojson = json.dumps({'type': 'FeatureCollection','features': [{'type': 'Feature','properties': {},'geometry': mapping(poly)}]})
        folium.GeoJson(geojson,style_function=lambda x: style_,tooltip='rank:'+str(rank)+' score:'+str(score)).add_to(m)
    
    # Fit map to the extent of topk-regions
    m.fit_bounds(coords)

    feature_group.add_to(m)
#    folium.LayerControl().add_to(m)

    return m


##################################################################################




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
    
        # compute 'init'
        categories = [G.nodes[n]['cat'] for n in region]
        init = [categories.count(t) for t in types]
        
        # compute score
        score, rel_se = compute_score(init, params)
    
        # update top-k regions
        topk_regions = update_topk_list(topk_regions, region_core, region_border, rel_se, score, init, params, start_time, updates)
#         if score > topk_regions[params['settings']['top_k'] - 1][0]:
#             topk_regions.append([score, rel_se,
#                                  [region_core.copy(), region_border.copy()],
#                                  init.copy(), len(region)])
#             topk_regions = sorted(topk_regions, key=lambda topk_regions: topk_regions[0], reverse=True)
#             topk_regions = topk_regions[:-1]
#             updates[time.time() - start_time] = topk_regions[params['settings']['top_k'] - 1][0]
    
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

    # compute 'init'
    categories = [G.nodes[n]['cat'] for n in new_region]
    init = [categories.count(t) for t in types]

    # compute score
    score, rel_se = compute_score(init, params)
    
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
#             if new_score > topk_regions[params['settings']['top_k'] - 1][0]:
#                 topk_regions.append([new_score,
#                                      new_rel_se,
#                                      [new_region_core.copy(), new_region_border.copy()],
#                                      init.copy(),
#                                      len(new_region)])
#                 topk_regions = sorted(topk_regions, key=lambda topk_regions: topk_regions[0], reverse=True)
#                 topk_regions = topk_regions[:-1]
#                 updates[time.time() - start_time] = topk_regions[params['settings']['top_k'] - 1][0]
                
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
#         if new_score > topk_regions[params['settings']['top_k'] - 1][0]:
#             topk_regions.append([new_score,
#                                  new_rel_se,
#                                  [new_region_core.copy(), new_region_border.copy()],
#                                  init.copy(),
#                                  len(new_region)])
#             topk_regions = sorted(topk_regions, key=lambda topk_regions: topk_regions[0], reverse=True)
#             topk_regions = topk_regions[:-1]
#             updates[time.time() - start_time] = topk_regions[params['settings']['top_k'] - 1][0]

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
            
            # compute 'init'
            categories = [G.nodes[n]['cat'] for n in region]
            init = [categories.count(t) for t in types]

            # compute score
            score, rel_se = compute_score(init, params)

            # update top-k regions
            topk_regions = update_topk_list(topk_regions, region_core, region_border, rel_se, score, init, params, start_time, updates)
#             if score > topk_regions[params['settings']['top_k'] - 1][0]:
#                 topk_regions.append([score, rel_se,
#                                      [region_core.copy(), region_border.copy()],
#                                      init.copy(), len(region)])
#                 topk_regions = sorted(topk_regions, key=lambda topk_regions: topk_regions[0], reverse=True)
#                 topk_regions = topk_regions[:-1]
#                 updates[time.time() - start_time] = topk_regions[params['settings']['top_k'] - 1][0]

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
        
            # compute 'init'
            categories = [G.nodes[n]['cat'] for n in region]
            init = [categories.count(t) for t in types]

            # compute score
            score, rel_se = compute_score(init, params)

            # update top-k regions
            topk_regions = update_topk_list(topk_regions, region_core, region_border, rel_se, score, init, params, start_time, updates)
#             if score > topk_regions[params['settings']['top_k'] - 1][0]:
#                 topk_regions.append([score, rel_se,
#                                      [region_core.copy(), region_border.copy()],
#                                      init.copy(), len(region)])
#                 topk_regions = sorted(topk_regions, key=lambda topk_regions: topk_regions[0], reverse=True)
#                 topk_regions = topk_regions[:-1]
#                 updates[time.time() - start_time] = topk_regions[params['settings']['top_k'] - 1][0]

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
#                 if new_score > topk_regions[params['settings']['top_k'] - 1][0]:
#                     topk_regions.append([new_score,
#                                          new_rel_se,
#                                          [new_region_core.copy(), new_region_border.copy()],
#                                          init.copy(),
#                                          len(new_region)])
#                     topk_regions = sorted(topk_regions, key=lambda topk_regions: topk_regions[0], reverse=True)
#                     topk_regions = topk_regions[:-1]
#                     updates[time.time() - start_time] = topk_regions[params['settings']['top_k'] - 1][0]

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


def show_map(gdf, region, colors):
    
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


def run_mbrs(gdf, G, rtree, types, params, start_time, seeds):
    """Computes the top-k high/low mixture regions.
    
    Args:
         gdf: A GeoDataFrame containing the input points.
         G: The spatial connectivity graph over the input points.
         rtree: The R-tree index constructed over the input points.
         types: The set of distinct point types.
         params: The configuration parameters.
         start_time: The starting time of the execution.
         seeds: The set of seeds to be used.
                  
    Returns:
         The list of top-k regions found within the given time budget.
    """
    
    # Initialize top-k list
    topk_regions = []
    while len(topk_regions) < params['settings']['top_k']:
        topk_regions.append([0, 0, [set(), set()], [], 0])   # [score, rel_se, [region_core, region_border], init, length]

    iterations = 0
    updates = dict()
    
    if params['methods']['current'] == 'ExpHybrid':
        topk_regions = run_exp_hybrid(G, seeds, params, types, topk_regions, start_time, updates)
    elif params['methods']['current'] == 'ExpCircles':
        topk_regions = run_exp_circles(gdf, rtree, G, seeds, params, types, topk_regions, start_time, updates)
    else:
        queue, topk_regions = init_queue(G, seeds, types, params, topk_regions, start_time, updates)

        # Process queue
        while (time.time() - start_time) < params['variables']['time_budget']['current'] and len(queue) > 0:
            iterations += 1
            score, topk_regions = process_queue(G, queue, topk_regions, params, types, start_time, updates)
            
    return topk_regions, updates

#    return topk_regions[0][0], topk_regions[0][2][0].union(topk_regions[0][2][1]), updates # return score, region and updates
