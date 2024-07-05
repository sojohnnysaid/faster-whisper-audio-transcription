import subprocess
import os
from faster_whisper import WhisperModel

def extract_audio(video_file, output_audio_file):
    print("Extracting audio from video...")
    subprocess.run(['ffmpeg', '-i', video_file, '-ac', '1', '-ar', '16000', output_audio_file], check=True)
    print(f"Audio extracted to {output_audio_file}")

def transcribe_audio(audio_file, transcription_file):
    print("Loading Whisper model...")
    whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

    print("Transcribing audio...")
    segments, info = whisper_model.transcribe(audio_file, beam_size=5)

    with open(transcription_file, 'w') as file:
        for segment in segments:
            file.write(f"{segment.text}\n")
    print(f"Transcription saved to {transcription_file}")

def find_first_mp4(directory):
    for file in os.listdir(directory):
        if file.endswith('.mp4'):
            return file
    return None

def main():
    video_file = find_first_mp4('.')
    if video_file is None:
        print("Error: No MP4 video file found.")
        return

    base_name = os.path.splitext(video_file)[0]
    output_audio_file = f"{base_name}.wav"
    transcription_dir = 'transcriptions'
    transcription_file = os.path.join(transcription_dir, f"{base_name}_transcription.txt")

    if not os.path.exists(transcription_dir):
        os.makedirs(transcription_dir)

    if not os.path.exists(output_audio_file):
        extract_audio(video_file, output_audio_file)

    transcribe_audio(output_audio_file, transcription_file)

if __name__ == '__main__':
    main()
