env >> /etc/environment;
apt install -y python3 python3-pip curl ffmpeg --no-install-recommends
curl -o /root/run.py https://raw.githubusercontent.com/Kesehet/vastapibkup/main/run.py &


curl https://ollama.ai/install.sh | sh
pip install ollama transformers pydub pyannote.audio pytube
pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate
pip install flash-attn --no-build-isolation



ollama serve &
ollama run llama3 &


python3 /root/run.py




