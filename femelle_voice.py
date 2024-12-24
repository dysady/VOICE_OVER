##################################################
#Prétraitement
import sounddevice as sd
import numpy as np
import librosa
import librosa.display

# Paramètres de capture audio
SAMPLE_RATE = 16000  # Taux d'échantillonnage
FRAME_SIZE = 1024    # Taille de chaque frame
HOP_LENGTH = 512     # Déplacement pour le spectrogramme Mel
N_MELS = 80          # Nombre de bandes Mel

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    # Convertir le flux en spectrogramme Mel
    audio = indata[:, 0]  # Canal unique
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    process_audio(mel_spectrogram_db)

# Fonction pour démarrer la capture en temps réel
def start_audio_stream():
    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, callback=audio_callback, blocksize=FRAME_SIZE):
        print("Audio stream started...")
        sd.sleep(10000)  # Laisser tourner pendant 10 secondes pour les tests

start_audio_stream()

##########################################
#Transformation du spectrogramme
import torch
from fastspeech2_model import FastSpeech2  # Supposons que ce soit un wrapper du modèle

# Charger le modèle FastSpeech 2
model_path = "fastspeech2_pretrained.pth"
fastspeech_model = FastSpeech2()
fastspeech_model.load_state_dict(torch.load(model_path))
fastspeech_model.eval()

def transform_spectrogram(input_mel):
    """
    Transforme le spectrogramme Mel masculin en féminin.
    """
    with torch.no_grad():
        input_tensor = torch.tensor(input_mel).unsqueeze(0)  # Ajouter batch dimension
        transformed_mel = fastspeech_model(input_tensor)
    return transformed_mel.squeeze(0).numpy()

##############################################
#Synthèse audio
from hifigan_model import HiFiGAN  # Supposons que ce soit un wrapper du modèle

# Charger le modèle HiFi-GAN
hifigan_path = "hifigan_pretrained.pth"
hifigan = HiFiGAN()
hifigan.load_state_dict(torch.load(hifigan_path))
hifigan.eval()


#opti
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fastspeech_model.to(device)
hifigan.to(device)

from torch.quantization import quantize_dynamic

#fastspeech_model = quantize_dynamic(fastspeech_model, {torch.nn.Linear}, dtype=torch.qint8)
#hifigan = quantize_dynamic(hifigan, {torch.nn.Linear}, dtype=torch.qint8)


def synthesize_audio(mel_spectrogram):
    """
    Génère un signal audio à partir du spectrogramme Mel transformé.
    """
    with torch.no_grad():
        mel_tensor = torch.tensor(mel_spectrogram).unsqueeze(0)  # Ajouter batch dimension
        audio = hifigan(mel_tensor)
    return audio.squeeze(0).numpy()

##############################################
#Boucle temps réel
import threading
from queue import Queue

# File d'attente pour les segments audio
audio_queue = Queue()

def process_audio(mel_spectrogram):
    # Transformer le spectrogramme
    transformed_mel = transform_spectrogram(mel_spectrogram)
    # Synthétiser l'audio
    audio_output = synthesize_audio(transformed_mel)
    # Lecture de l'audio (thread séparé)
    threading.Thread(target=play_audio, args=(audio_output,)).start()

def play_audio(audio):
    """
    Joue le signal audio.
    """
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()

# Intégrer le tout
if __name__ == "__main__":
    start_audio_stream()




