#!/bin/sh
cd tests
exec python3 -m unittest discover "$@"
