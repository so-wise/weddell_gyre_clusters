# syntax=docker/dockerfile:1

# indicate the base image from this repository and tag
FROM pangeo/pangeo-notebook:2021.09.08

# copy the requirements file
COPY requirements.txt requirements.txt

# install the required packages
RUN pip install -r requirements.txt

# take the files and folders in your local build context and
# add them to the Docker image’s current working directory
COPY . .
