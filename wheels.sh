#!/bin/bash

# Create a directory for the downloaded wheels
mkdir -p wheels

# Base URL for the wheel files
BASE_URL="https://transpitch.com/python/wheel/"

# Read the list of files from wheels.txt and download each using curl in the background
while IFS= read -r file; do
    # Download each file using curl in the background
    curl -o "wheels/${file}" "${BASE_URL}${file}" &
done < wheels.txt

# Wait for all background jobs to complete
wait
