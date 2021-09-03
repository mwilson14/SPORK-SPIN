# SPORK-SPIN
SPORK...but with an attempt at implementing a mesocyclone detection algorithm

This is the github repo for the Supercell Polarimetric Observation Research Kit (SPORK), an automated Python algorithm for identifying and quantifying supercell dual-pol signatures in WSR-88D data. Metrics currently calculated by SPORK include:

ZDR arc size and intensity

ZDR column size and depth

KDP-ZDR separation angle and distance

Inferred hailfall area

Code and data for a soon-to-be-submitted paper describing SPORK and applying it to a large sample of supercells will also be stored on this repository.

This version of SPORK contains a preliminary attempt at adding a mesocyclone detection algorithm to the code. However, although we're reasonably confident that this algorithm can identify mesocyclones from observing plenty of its output, little quantitative verification of its estimates of mesocyclone strength has been performed, so it should be used with caution (or just ignored). Working with the velocity data slows down the algorithm a bit as well, so a streamlined version which does not have the attempt at a mesocyclone detection algorithm will soon be available.
