env >> /etc/environment;
apt install -y python3 htop python3-pip curl ffmpeg python3.10-venv --no-install-recommends

curl -o /root/run.py https://raw.githubusercontent.com/Kesehet/vastapibkup/main/run.py &
curl -o ollama.sh https://raw.githubusercontent.com/Kesehet/vastapibkup/main/ollama.sh
curl -O https://transpitch.com/python/requirements.txt
curl -O https://transpitch.com/python/all_requirements.txt
curl -O https://raw.githubusercontent.com/Kesehet/vastapibkup/main/wheels.sh
curl -O https://raw.githubusercontent.com/Kesehet/vastapibkup/main/wheels.txt

sed -i 's/\r//g' *.sh
chmod +x *.sh


./ollama.sh &
./wheels.sh

pip install -r all_requirements.txt --no-index --find-links /wheels
pip install -r requirements.txt


# pip install ollama transformers pydub pyannote.audio pytube
# pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate
# pip install flash-attn --no-build-isolation





python3 /root/run.py >> run.log 2>&1





