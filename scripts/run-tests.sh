#!/bin/bash

echo "Running tests..."

# Run the unittests tests
python -m unittest discover -s tests/data_processing

# $? stores exit value of the last command
if [ $? -ne 0 ]; then
    echo "Some tests failed!"
    exit 1
fi