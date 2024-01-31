#!/usr/bin/env bash

# Download and compile LKH Solver

VERSION=3.0.6

# Unpacking
tar xvfz LKH-$VERSION.tgz

# Making
cd LKH-$VERSION
make -j 4
