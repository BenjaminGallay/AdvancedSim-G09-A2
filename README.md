# What is this project about?

This repository contains the work for the second assignment of the Advanced Simulation course of TU Delft by EPA133a Group 09

## Goals of this project

This assignment focuses on the simulation of road traffic on the main roads of Bangladesh, using python and mesa. We aim at simulating the delays caused by the bridges being damaged, and we identify the bridges that cause the most delays and should be invested in first.

## Files

- the ```/data/``` folder contains the cleaned dataset from the teachers and the code preparing it to be fed into the simulation. More information in the ```/data/README.md``` file
- the ```/experiment/``` folder contains the output data of the simulation. Each .csv file contains the travel time and the id of every truck that reached the sink at the end of the road, in Dhaka.
- the ```/model/``` folder contains the code relative to the simulation and the display in the browser. More information in the ```/model/README.md``` file
- the ```/report/``` folder contains the report

## Dependencies 

- Mesa 2.1.4
- Pandas
- Numpy
- Openpyxl

## How to run ?

Simply execute ```$ python /model/model_run.py``` and follow the instructions in the console. More information about the code can be found in the README in each folder.
