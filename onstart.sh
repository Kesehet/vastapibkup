#!/bin/bash

log_time() {
    task_name=$1
    start_time=$(date +%s)
    
    shift
    "$@"
    
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    
    echo "${task_name}: ${elapsed_time} seconds" >> benchmark.log
}

log_time "Setting environment variables" env >> /etc/environment
log_time "Installing packages" apt install -y python3 htop python3-pip curl ffmpeg python3.10-venv --no-install-recommends

log_time "Downloading run.py" curl -o /root/run.py https://raw.githubusercontent.com/Kesehet/vastapibkup/main/run.py &
log_time "Downloading ollama.sh" curl -o ollama.sh https://raw.githubusercontent.com/Kesehet/vastapibkup/main/ollama.sh
log_time "Downloading requirements.txt" curl -O https://transpitch.com/python/requirements.txt
log_time "Downloading all_requirements.txt" curl -O https://transpitch.com/python/all_requirements.txt
log_time "Downloading wheels.sh" curl -O https://raw.githubusercontent.com/Kesehet/vastapibkup/main/wheels.sh
log_time "Downloading wheels.txt" curl -O https://raw.githubusercontent.com/Kesehet/vastapibkup/main/wheels.txt

log_time "Removing carriage returns from .sh files" sed -i 's/\r//g' *.sh
log_time "Setting execute permission on .sh files" chmod +x *.sh

log_time "Executing ollama.sh" ./ollama.sh &
log_time "Executing wheels.sh" ./wheels.sh

log_time "Installing Python packages from all_requirements.txt" pip install -r all_requirements.txt --no-index --find-links /wheels
log_time "Installing Python packages from requirements.txt" pip install -r requirements.txt

# The commented lines can also be timed if they are uncommented and used
# log_time "Installing additional Python packages" pip install ollama transformers pydub pyannote.audio pytube
# log_time "Upgrading transformers and accelerate" pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate
# log_time "Installing flash-attn" pip install flash-attn --no-build-isolation

log_time "Running run.py script" python3 /root/run.py >> run.log 2>&1
