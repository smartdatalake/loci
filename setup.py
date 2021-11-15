from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
   name='loci-st',
   version='0.1',
   description='LOCI is a component providing functionalities for analysing, mining, and visualizing spatial and temporal data.',
   long_description=long_description,
   long_description_content_type="text/markdown",
   author='Alexandros Zeakis, Dimitrios Skoutas, Kostas Patroumpas, Georgios Chatzigeorgakidis, Panagiotis Kalampokis',
   author_email='azeakis@athenarc.gr, dskoutas@athenarc.gr, kpatro@athenarc.gr, gchatzi@athenarc.gr, pkalampokis@athenarc.gr',
   packages=['loci'],
   url='https://github.com/smartdatalake/loci',
   python_requires='>=3.6', 
   install_requires = ["Cython==0.29.10", "wheel==0.33.4",
                       "pandas==1.2.0", "geopandas==0.8.1", "mlxtend==0.18.0", 
                       "Rtree==0.9.7", "networkx==2.5.1", "folium==0.11.0",
                       "ipycytoscape==1.2.2", "wordcloud==1.8.1", "osmnx==1.0.0",
                       "plotly==4.14.0", "seaborn==0.11.2", "statsmodels==0.9.0",
                       "ruptures==1.1.5", "voila==0.2.10"]
)
