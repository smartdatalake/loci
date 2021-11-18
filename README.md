## LOCI

#### Overview

LOCI is a Python library offering various functions for analysing, mining, and visualizing spatial and temporal data. Specifically, it provides functionalities for spatial exploration and mining over Points and Areas of Interest, as well as for change detection and seasonality decomposition in time series data, and evolution tracking of dynamic sets of entities. LOCI is under active development, originally started in the EU H2020 project [SLIPO](http://slipo.eu/) and being currently extended in the EU H2020 project [SmartDataLake](https://smartdatalake.eu/) to include: (i) detection of Areas of Interest characterized by high or low mixture of POI types, and (ii) analytics over time series data and evolving sets of entities.

#### Quick start

Please see the provided [notebooks](https://github.com/smartdatalake/loci/tree/master/notebooks).

#### Documentation

Please see [here](https://smartdatalake.github.io/loci/).

### Installation

#### Python Module

LOCI can be found [here](https://pypi.org/project/loci-st/) and installed with:

```sh
$ pip install loci-st
```

#### Creating and launching a Docker image 

We provide a `Dockerfile` that may be used to create a Docker image (`loci_st`) from the executable:

```sh
$ docker build -t smartdatalake/loci .
```

This docker image can then be used to launch a web service application via voila as follows:

```sh
$ docker run smartdatalake/loci
```

The image can also be found [here](https://hub.docker.com/r/smartdatalake/loci) and pulled with:

```sh
$ docker pull smartdatalake/loci
```

#### License

The contents of this project are licensed under the [Apache License 2.0](https://github.com/smartdatalake/loci/blob/master/LICENSE).
