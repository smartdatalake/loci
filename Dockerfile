FROM python:3.7
ADD . /
RUN pip install loci-st
ENTRYPOINT [ "voila", "./notebooks/Demo-App.ipynb" ]
