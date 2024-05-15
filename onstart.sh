env >> /etc/environment;
apt install -y python3 htop python3-pip curl ffmpeg python3.10-venv --no-install-recommends
curl -o /root/run.py https://raw.githubusercontent.com/Kesehet/vastapibkup/main/run.py &

curl -o ollama.sh https://raw.githubusercontent.com/Kesehet/vastapibkup/main/ollama.sh
sed -i 's/\r//g' ollama.sh
chmod +x ollama.sh
./ollama.sh &

python3 -m venv myenv
source myenv/bin/activate


pip install ollama transformers pydub pyannote.audio pytube
pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate
pip install flash-attn --no-build-isolation


pip freeze > requirements.txt
pip wheel -r requirements.txt -w wheels &





python3 /root/run.py >> run.log 2>&1





