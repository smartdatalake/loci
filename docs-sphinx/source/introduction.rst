Introduction
============


LOCI is a suite of tools for analysing, mining, and visualizing spatial and temporal data. It offers functionalities for spatial exploration and mining over Points and Areas of Interest, as well as for change detection and seasonality decomposition in time series data, and evolution tracking of dynamic sets of entities. Specifically, the main functionalities that are currently included in LOCI are outlined below.

Spatial
-------

LOCI provides high-level functions for exploring, mining and visualizing Points and Areas of Interest. More specifically, it offers capabilities for:

* *Keyword-based exploration*:

  * Filter POIs by category or keywords.
  * Compute and visualize statistics about POI categories or keywords.
  * Generate `word clouds <http://amueller.github.io/word_cloud/>`_ from POI categories or keywords.
  
* *Map-based exploration* using `folium <https://github.com/python-visualization/folium>`_ functionalities:

  * Display POIs on the map.
  * Generate heatmaps from POI locations.
  * Construct and visualize a grid structure over POI locations.
  
* *Discovery of Areas of Interest*:

  * Compute and visualize POI clusters using density-based clustering, in particular using the `DBSCAN <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_ and `HDBSCAN <https://hdbscan.readthedocs.io/en/latest/index.html>`_ algorithms.
  * Employ Latent Dirichlet Allocation (`LDA <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition>`_) to extract and assign topics to clusters. 
  * Use frequent itemset mining to discover frequent location sets.
  * Detect `mixture-based regions of arbitrary shapes <https://dl.acm.org/doi/abs/10.1145/3474717.3484215>`_ that exhibit either very high or very low diversity in the types of POIs therein.

Temporal
--------

This component provides various analytics for time series data, either on a single or a set of sequences. More specifically, it includes the following functionalities:

* **Single time series** 

  * *Seasonal decomposition*: Analyzes and visualizes the trend, seasonality and residual components of a time series, and selects the most appropriate period among a given set of periods (using the `seasonal_decompose <https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.seasonal_decompose.htm>`_ algorithm).
  * *Change detection*: Detects abrupt changes in a given time series (using the `Pelt <https://github.com/deepcharles/ruptures>`_ algorithm).

* **Multiple time series**

  * *SankeyTS*: Given a set of time series, it generates an interactive visualization based on a Sankey diagram.
  * *Change detection*: Detects collective changes within a set of time series, i.e., groups of time series that abruptly change during the same time period (using the `Pelt <https://github.com/deepcharles/ruptures>`_ algorithm).

* **Set Evolution**

  * *CDEvSet*: This component tracks and analyses changes in groups of entities that evolve over time.
