env >> /etc/environment;
apt install python3 -y
apt install python3-pip -y
apt install curl -y
apt install ffmpeg -y

curl https://ollama.ai/install.sh | sh
pip install ollama transformers pydub pyannote.audio pytube
pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]
pip install flash-attn --no-build-isolation

ollama serve &
ollama run openchat &




curl -o /root/run.py https://raw.githubusercontent.com/Kesehet/vastapibkup/main/run.py

python3 /root/run.py
