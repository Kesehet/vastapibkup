#!/bin/bash

# Base URL for the wheel files
BASE_URL="https://transpitch.com/python/wheel/"

# Read the list of files from wheels.txt
while IFS= read -r file; do
    # Download each file using aria2
    aria2c -x 16 -s 16 -k 1M "${BASE_URL}${file}" -o "${file}"
done < wheels.txt
