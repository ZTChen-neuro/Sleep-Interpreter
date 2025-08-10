#!/bin/bash

all: test

test:
	cd ./layers && make test
	python naive_transformer.py

clean:
	rm -rf ./__pycache__
	cd ./layers && make clean

