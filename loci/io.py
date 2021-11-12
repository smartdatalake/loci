import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.geometry import Polygon
import geopandas as gpd
import math
import osmnx
import requests
from io import BytesIO
from zipfile import ZipFile


def read_csv(input_file, sep=',', col_id='id', col_name='name', col_lon='lon', col_lat='lat', col_kwds='keywords', kwds_sep=';', source_crs='EPSG:4326', target_crs='EPSG:4326'):
    """Create a DataFrame from a CSV file and then convert to GeoDataFrame.
    
    Args:
        input_file (string): Path to the input CSV file.
        sep (string): Column delimiter (default: `;`).
        col_id (string): Name of the column containing the id (default: `id`).        
        col_name (string): Name of the column containing the name (default: `name`).        
        col_lon (string): Name of the column containing the longitude (default: `lon`).
        col_lat (string): Name of the column containing the latitude (default: `lat`).
        col_kwds (string): Name of the column containing the keywords (default: `kwds`).
        kwds_sep (string): Keywords delimiter (default: `;`).
        source_crs (string): Coordinate Reference System of input data (default: `EPSG:4326`).
        target_crs (string): Coordinate Reference System of the GeoDataFrame to be created (default: `EPSG:4326`).
        
    Returns:
        A GeoDataFrame.
    """
    
    df = pd.read_csv(input_file, sep=sep, error_bad_lines=False)
    df = df.rename(columns={col_id: 'id', col_name: 'name', col_lon: 'lon', col_lat: 'lat', col_kwds: 'kwds'})
    df['id'].replace('', np.nan, inplace=True)
    df.dropna(subset=['id'], inplace=True)
    df['name'].replace('', np.nan, inplace=True)
    df.dropna(subset=['name'], inplace=True)
    df['kwds'].replace('', np.nan, inplace=True)
    df.dropna(subset=['kwds'], inplace=True)       
    df = df[pd.to_numeric(df['lon'], errors='coerce').notnull()]
    df = df[pd.to_numeric(df['lat'], errors='coerce').notnull()]
    df['lon'] = df['lon'].apply(lambda x: float(x))
    df['lat'] = df['lat'].apply(lambda x: float(x))
    df['kwds'] = df['kwds'].apply(lambda x: x.split(kwds_sep))
    
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
    gdf.drop(['lon', 'lat'], inplace=True, axis=1)
    gdf = gdf.set_crs(source_crs)
    if target_crs != source_crs:
        gdf = gdf.to_crs(target_crs)
        
    return gdf


def crop(gdf, min_lon, min_lat, max_lon, max_lat):
    """Crops the given GeoDataFrame according to the given bounding box.
    
    Args:
        gdf (GeoDataFrame): The original GeoDataFrame.
        min_lon, min_lat, max_lon, max_lat (floats): The bounds.
        
    Returns:
        The cropped GeoDataFrame.
    """
    
    polygon = Polygon([(min_lon, min_lat),
                       (min_lon, max_lat),
                       (max_lon, max_lat),
                       (max_lon, min_lat),
                       (min_lon, min_lat)])
    return gpd.clip(gdf, polygon)


def read_poi_csv(input_file, col_id='id', col_name='name', col_lon='lon', col_lat='lat', col_kwds='kwds', col_sep=';',
                 kwds_sep=',', source_crs='EPSG:4326', target_crs='EPSG:4326', keep_other_cols=False):
    """Creates a POI GeoDataFrame from an input CSV file.

    Args:
        input_file (string): Path to the input csv file.
        col_id (string): Name of the column containing the POI id (default: `id`).
        col_name (string): Name of the column containing the POI name (default: `name`).
        col_lon (string): Name of the column containing the POI longitude (default: `lon`).
        col_lat (string): Name of the column containing the POI latitude (default: `lat`).
        col_kwds (string): Name of the column containing the POI keywords (default: `kwds`).
        col_sep (string): Column delimiter (default: `;`).
        kwds_sep (string): Keywords delimiter (default: `,`).
        source_crs (string): Coordinate Reference System of input data (default: `EPSG:4326`).
        target_crs (string): Coordinate Reference System of the GeoDataFrame to be created (default: `EPSG:4326`).
        keep_other_cols (bool): Whether to keep the rest of the columns in the csv file (default: `False`).

    Returns:
        A POI GeoDataFrame with columns `id`, `name` and `kwds`.
    """

    def lon_lat_to_point(row, c_lon, c_lat):
        try:
            x_lon = float(row[c_lon])
            y_lat = float(row[c_lat])
            if math.isnan(x_lon) is False and math.isnan(y_lat) is False:
                return Point(x_lon, y_lat)
            else:
                return float('NaN')
        except:
            return float('NaN')

    pois = pd.read_csv(input_file, delimiter=col_sep, error_bad_lines=False)
    init_poi_size = pois.index.size

    columns = list(pois)
    subset_cols = []

    # Columns to Check for N/A, Nulls
    if keep_other_cols:
        subset_cols.extend(columns)
    else:
        subset_cols = [col_id, col_lon, col_lat]
        if col_name in columns:
            subset_cols.append(col_name)
        if col_kwds in columns:
            subset_cols.append(col_kwds)

    # Geometry Column(Uncleaned)
    pois['geometry'] = pois.apply(lambda row: lon_lat_to_point(row, col_lon, col_lat), axis=1)
    subset_cols.append('geometry')

    # Drop Columns Not in subset Columns.
    drop_columns = set(columns) - set(subset_cols)
    pois.drop(drop_columns, inplace=True, axis=1)

    # Drop all N/A, Null rows from DataFrame.
    pois.dropna(inplace=True)
    if init_poi_size - pois.index.size > 0:
        print("Skipped", (init_poi_size - pois.index.size), "rows due to errors.")

    if col_kwds in columns:
        pois[col_kwds] = pois[col_kwds].map(lambda s: s.split(kwds_sep))

    source_crs = {'init': source_crs}
    target_crs = {'init': target_crs}
    pois = gpd.GeoDataFrame(pois, crs=source_crs, geometry=pois['geometry']).to_crs(target_crs).drop(columns=[col_lon,
                                                                                                              col_lat])

    print('Loaded ' + str(len(pois.index)) + ' POIs.')

    return pois


def import_osmnx(bound, tags, target_crs='EPSG:4326'):
    """Creates a POI GeoDataFrame from POIs retrieved by OSMNX (https://github.com/gboeing/osmnx).

    Args:
        bound (polygon): A polygon to be used as filter.
        tags (dict): A dictionary of tags regarding POIs in order to filter OSM entities. For example, tags = {'building': True} would return all building footprints in the area; tags = {'amenity':True, 'landuse':['retail','commercial'], 'highway':'bus_stop'} would return the specific amenities.
        target_crs (string): Coordinate Reference System of the GeoDataFrame to be created (default: `EPSG:4326`).

    Returns:
        A POI GeoDataFrame with columns `id`, `name` and `kwds`.
    """

    # retrieve pois
    pois = osmnx.geometries.geometries_from_polygon(bound, tags)

    if len(pois.index) > 0:
        # filter pois
        pois = pois[pois.amenity.notnull()]
        pois_filter = pois.element_type == 'node'
        pois = pois[pois_filter]

        # restructure gdf
        subset_cols = ['osmid', 'amenity', 'name', 'geometry']
        columns = list(pois)
        drop_columns = set(columns) - set(subset_cols)
        pois.drop(drop_columns, inplace=True, axis=1)
        pois = pois.reset_index(drop=True)
        pois = pois.rename(columns={'osmid': 'id', 'amenity': 'kwds'})
        pois['kwds'] = pois['kwds'].map(lambda s: [s])

    if target_crs != 'EPSG:4326':
        target_crs = {'init': target_crs}
        pois = pois.to_crs(target_crs)

    print('Loaded ' + str(len(pois.index)) + ' POIs.')

    return pois


def import_osmwrangle(osmwrangle_file, target_crs='EPSG:4326', bound=None):
    """Creates a POI GeoDataFrame from a file produced by OSMWrangle (https://github.com/SLIPO-EU/OSMWrangle).

    Args:
        osmwrangle_file (string): Path or URL to the input csv file.
        target_crs (string): Coordinate Reference System of the GeoDataFrame to be created (default: `EPSG:4326`).
        bound (polygon): A polygon to be used as filter.

    Returns:
        A POI GeoDataFrame with columns `id`, `name` and `kwds`.
    """

    def lon_lat_to_point(row, c_lon, c_lat):
        x_lon = float(row[c_lon])
        y_lat = float(row[c_lat])
        if math.isnan(x_lon) is False and math.isnan(y_lat) is False:
            return Point(x_lon, y_lat)
        else:
            return float('NaN')

    col_sep = '|'
    col_id = 'ID'
    col_lon = 'LON'
    col_lat = 'LAT'
    col_name = 'NAME'
    col_cat = 'CATEGORY'
    col_subcat = 'SUBCATEGORY'
    source_crs = {'init': 'EPSG:4326'}

    # Load the file
    if osmwrangle_file.startswith('http') and osmwrangle_file.endswith('.zip'):
        response = requests.get(osmwrangle_file)
        zip_file = ZipFile(BytesIO(response.content))
        with zip_file.open(zip_file.namelist()[0]) as csvfile:
            pois = pd.read_csv(csvfile, delimiter=col_sep, error_bad_lines=False)
    else:
        pois = pd.read_csv(osmwrangle_file, delimiter=col_sep, error_bad_lines=False)

    init_poi_size = pois.index.size

    columns = list(pois)

    subset_cols = [col_id, col_name, 'kwds', col_lon, col_lat]

    # Geometry Column(Uncleaned)
    pois['geometry'] = pois.apply(lambda row: lon_lat_to_point(row, col_lon, col_lat), axis=1)
    subset_cols.append('geometry')

    pois['kwds'] = pois[col_cat] + ',' + pois[col_subcat]
    pois['kwds'] = pois['kwds'].map(lambda s: s.split(','))

    # Drop Columns Not in subset Columns.
    drop_columns = set(columns) - set(subset_cols)
    pois.drop(drop_columns, inplace=True, axis=1)

    # Drop all N/A, Null rows from DataFrame.
    pois.dropna(inplace=True)
    if init_poi_size - pois.index.size > 0:
        print("Skipped", (init_poi_size - pois.index.size), "rows due to errors.")

    pois = pois.rename(columns={col_id: 'id', col_name: 'name'})
    pois = gpd.GeoDataFrame(pois, crs=source_crs, geometry=pois['geometry']).drop(columns=[col_lon, col_lat])

    # Check whether location filter should be applied
    if bound is not None:
        spatial_filter = pois.geometry.intersects(bound)
        pois = pois[spatial_filter]

    if target_crs != 'EPSG:4326':
        target_crs = {'init': target_crs}
        pois = pois.to_crs(target_crs)

    print('Loaded ' + str(len(pois.index)) + ' POIs.')

    return pois


def retrieve_osm_loc(name, buffer_dist=0):
    """Retrieves a polygon from an OSM location.

    Args:
         name (string): Name of the location to be resolved.
         buffer_dist (numeric): Buffer distance in meters.

    Returns:
        A polygon.
    """

    geom = osmnx.geocode_to_gdf(name, buffer_dist=buffer_dist)
#    geom = osmnx.core.gdf_from_place(name, buffer_dist=buffer_dist)
    if len(geom.index) > 0:
        geom = geom.iloc[0].geometry
    else:
        geom = None

    return geom


def to_geojson(gdf, output_file):
    """Exports a GeoDataFrame to a GeoJSON file.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame object to be exported.
        output_file (string): Path to the output file.
    """

    gdf.to_file(output_file, driver='GeoJSON')
