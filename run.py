
MODEL = "llama3"
import ollama
import time
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import urllib
import os
import uuid
import json
import asyncio
from threading import Lock
import sys
import traceback
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pytubefix import YouTube
import requests



def ollama_ask(question):
    if isinstance(question, str):
        response = ollama.chat(model=MODEL, messages=[
            {'role': 'user', 'content': question}
        ])
    elif isinstance(question, list):
        response = ollama.chat(model=MODEL, messages=question)
    print(question)

    return response.get("message",{}).get("content")



def get_filename_without_extension(file_path):
    filename_without_extension, _ = os.path.splitext(file_path)
    return filename_without_extension


def seconds_to_timestamp(seconds):
    # Convert seconds to hours, minutes, and seconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    # Format the timestamp
    timestamp = f"{hours:02}:{minutes:02}:{seconds:02}"

    return timestamp



def is_youtube_url(url):
    """
    Checks if the given URL is a valid YouTube video URL.

    :param url: URL to check
    :return: True if URL is a YouTube video, False otherwise
    """
    youtube_domains = ['youtube.com', 'www.youtube.com', 'youtu.be', 'www.youtu.be']
    if any(domain in url for domain in youtube_domains):
        return True
    return False

def download_youtube_video(url, save_path='.'):
    try:
        print(f"Downloading from youtube ... {url}")
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        file_path = stream.download(output_path=save_path)  # Capture the file path of the downloaded video
        file_name = os.path.basename(file_path)  # Extract the file name from the file path
        print(f"Video '{yt.title}' has been downloaded successfully as '{file_name}'.")
        return file_name  # Return the file name instead of the video title
    except Exception as e:
        print(e)
        return "Error"

def download_file(url):
    try:
        if is_youtube_url(url):
          print("Downloading from youtube ... ")
          return download_youtube_video(url)
        response = requests.get(url)
        if response.status_code == 200:
            filename = url.split('/')[-1]
            with open(filename, 'wb') as file:
                file.write(response.content)
            return filename
        else:
            return "Error: Failed to download the file, status code {}".format(response.status_code)
    except Exception as e:
        return "Error: An exception occurred - {}".format(e)



def writeToFile(text,file):
  with open(file, 'w', encoding='utf-8') as f:
    f.write(text)



def getSpeakersData(sample, min_duration=0.5):
    waveform, sample_rate = torchaudio.load(sample)
    speakersData = pyannote_pipeline({"waveform": waveform, "sample_rate": sample_rate})
    result = []

    for turn, _, speaker in speakersData.itertracks(yield_label=True):
      result.append({
          "start": turn.start,
          "end": turn.end,
          "speaker": speaker,
          #"audio": cropAudio(turn.start, turn.end, sample)
      })

    return result

def get_matched_texts(result, Speakers):
    matched_texts = []
    for text in result["chunks"]:
        start, end = text['timestamp']
        Speaker_Now = ""
        if start is None or end is None:
            continue
        for speaker in Speakers:
            # Check if the speaker's time range includes the text's timestamp
            if round(float(speaker["start"]), 1) <= start and round(float(speaker["end"]), 1) >= end:
                Speaker_Now = speaker["speaker"]
                break  # Exit the loop once the matching speaker is found
        matched_texts.append({
            "text": text["text"],
            "start": start,
            "end": end,
            "speaker": Speaker_Now,
            "timestamp": text["timestamp"]
        })
    return matched_texts

def cropAudio(start,end,sample):
  directory = get_filename_without_extension(sample)
  audio = AudioSegment.from_file(sample)
  if not os.path.exists(directory):
    os.makedirs(directory)

  start_ms = start * 1000
  end_ms = end * 1000
  segment = audio[start_ms:end_ms]

  segment_path = os.path.join(directory, f"chunk_s{start_ms}_e{end_ms}.mp3")

  segment.export(segment_path,format="mp3")
  return segment_path




device = "cuda:0" if torch.cuda.is_available() else "cpu"
#Creating the pyannote pipeline
pyannote_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")
pyannote_pipeline.to(torch.device(device))
# ______________________________________________



torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)










def whisper_run(url, task_type="Translate", params={},context=""):
  try:
    print("Trying to download...")
    # Assume download_file(url) downloads a file and returns its filename
    filename = download_file(url)
    if filename == None or filename.find("age restriction") > 0:
        return {
            "error": filename,
            "duration": 0,
            "summary" : filename,
            "chunks": []
        }
    print(f"downloaded {filename}")
    # Load the file (pydub automatically detects its format)
    audio = AudioSegment.from_file(filename)
    audio_length_minutes = audio.duration_seconds / 60
    # Construct the new filename with .wav extension
    base_name = os.path.splitext(filename)[0]  # Remove the original extension
    new_filename = f"{base_name}.wav"

    # Export to WAV format using the new filename
    audio.export(new_filename, format="wav")

    # Update the filename variable with the new filename
    filename = new_filename


    res = ""
    response = { 'summary':'' , 'chunks':'', "duration":audio_length_minutes }
    # Ideal string is Translate&Transcribe&Summarize&Speaker Diarazation
    # response should be { 'summary':'' , 'chunks':''  }
    if params["language"] == None:
        params["language"] = "English"
    if(task_type == "Translate"):
        res = whisper_pipe(filename,return_timestamps=True,generate_kwargs=params)
        response["summary"] = res['text']
        response["chunks"] = res["chunks"]



    elif(task_type == "Transcribe"):
        res = whisper_pipe(filename,return_timestamps=True,generate_kwargs=params)
        response["summary"] = res["text"]
        response["chunks"] = res["chunks"]



    
    elif(task_type == "Paraphrase"):
        params["task"] = "transcribe"
        res = whisper_pipe(filename,return_timestamps=True,generate_kwargs=params)
        ctx = context if context != "" else "Summarize this ... "
        print("\n\n\n\n\n\n\n\n\n\n\n\n"+ctx+"\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        response["summary"] = ollama_ask( ctx + res['text'] )
        response["chunks"] = res["chunks"]
      


    elif(task_type == "Speaker Diarization"):
        res = get_matched_texts(
            whisper_pipe(filename,return_timestamps=True,generate_kwargs=params),
            getSpeakersData(filename)
        )
        print(res)
        response["summary"] = ""
        response["chunks"] = res



    elif(task_type == "Translate&Transcribe"):
        params["task"] = "transcribe"
        res1 = whisper_pipe(filename,return_timestamps=True,generate_kwargs=params)
        params["task"] = "translate"
        res2 = whisper_pipe(filename,return_timestamps=True,generate_kwargs=params)
        response["summary"] = f'''
            <h4>Translation</h4>
            {res1["text"]}
            <br/>
            <h4>Transcription</h4>
            {res2["text"]}
            <br/>
        '''
        retChunks = []
        for chunk in res1["chunks"]:
            for chunk2 in res2["chunks"]:
                if chunk["timestamp"] == chunk2["timestamp"]:

                    retChunks.append({
                        "translation": chunk2["text"],
                        "transcription": chunk["text"],
                        "timestamp": chunk["timestamp"]
                    })
        response["chunks"] = retChunks
    return response
  except Exception as e:
    print("We ran into error.")
    print(e)
    return e







import requests
import time

def fetch_llm_tasks():
    url = "https://llm.mediapitch.in/api/tasks/pending/llm"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
        data = response.json()

        if data:
            # Task is available
            print("Task available:", data)
            update_task(data["uuid"],"completed",ollama_ask(data["payload"]["query"]))
            return data
        else:
            # No task available
            print("No task available at the moment.")
            return {}
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")


def fetch_whisper_tasks():
    url = "https://llm.mediapitch.in/api/tasks/pending/whisper"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
        data = response.json()


        if data:
            # Task is available
            print("Task available:", data)
            try:
                print("Attempting whisper")
                do = whisper_run(data["payload"]["audio_url"],data["payload"]["task"], data["payload"]["params"],data["payload"]["context"] )
                print("whisper done")
                update_task(data["uuid"],"completed",do)
                print("task updated...")
                return data
            except Exception as e:
                update_task(data["uuid"],"failed",{"error":str(e)})
        else:
            # No task available
            print("No task available at the moment.")
            return {}
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        update_task(data["uuid"],"failed",{"error":str(http_err)})
    except Exception as err:
        print(f"An error occurred: {err}")


def update_task(uuid, status, result=None):
    url = f"https://llm.mediapitch.in/api/tasks/{uuid}"

    # Prepare the data payload for the PATCH request
    payload = {'status': status}
    if result is not None:
        payload['result'] = result

    try:
        response = requests.patch(url, json=payload)
        response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
        updated_task = response.json()

        print("Task updated successfully:", updated_task)
        return updated_task
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")

while 1:
    fetch_llm_tasks()
    fetch_whisper_tasks()
    time.sleep(2)

