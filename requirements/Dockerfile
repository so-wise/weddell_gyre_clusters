# syntax=docker/dockerfile:1
FROM pangeo/pangeo-notebook:2021.07.17

#WORKDIR /code
#ENV FLASK_APP=app.py
#ENV FLASK_RUN_HOST=0.0.0.0

RUN conda install -n notebook -c conda-forge <somepackage> && conda clean -tipsy

# COPY requirements.txt requirements.txt
# RUN pip install -r requirements.txt
#EXPOSE 5000
#COPY . .
#CMD ["flask", "run"]



