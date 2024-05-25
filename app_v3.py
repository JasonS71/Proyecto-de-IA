import torch
import torchaudio
import numpy as np
import scipy
import youtube_dl
import stempeg
import os
from IPython.display import Audio, display
from openunmix import predict
from IPython.display import HTML
from torch import nn, optim
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import textwrap
from pydub import AudioSegment
import whisper
from whisper import load_model, transcribe
import soundfile as sf


# ----------------------------------------------------------------------------------------------------------
# Entrada

audio_path = "Dataset de Prueba/1. Calle 13   Latinoamérica.mp3"
# ----------------------------------------------------------------------------------------------------------

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# ---------------------------------------------------

# Leer el archivo de audio
audio = AudioSegment.from_file(audio_path)
duration = len(audio) / 1000
stop_test = int(duration)

start = 0
stop = stop_test

# Leer el archivo de audio
audio_stems, rate = stempeg.read_stems(
    audio_path,
    sample_rate=44100,
    start=start,
    duration=(stop - start),
)

# Asegurarse de que los datos de audio sean válidos
audio_stems = np.nan_to_num(audio_stems)  # Reemplazar NaN por 0 y infinitos por valores finitos grandes

# Escalar los datos de audio al rango [-1, 1]
max_val = np.max(np.abs(audio_stems))
if max_val > 0:
    audio_stems = audio_stems / max_val

print("Fase 1 = Completada (Lectura del archivo de audio)")

# Realizar la separación
try:
    estimates = predict.separate(
        audio=torch.as_tensor(audio_stems).float(),
        rate=44100,
        targets=['vocals'],
        residual=True,
        device=device
    )
except Exception as e:
    print(f"Error during separation: {e}")
    raise

# Mostrar las estimaciones de las fuentes separadas
for target, estimate in estimates.items():
    # print(target)
    
    # Convertir a numpy y verificar valores NaN e Inf
    estimate_np = estimate.detach().cpu().numpy()[0]
    
    # Asegurarse de que no haya valores NaN o infinitos en la estimación
    estimate_np = np.nan_to_num(estimate_np)  # Reemplazar NaN por 0 y infinitos por valores finitos grandes
    
    # Normalizar el rango de -1 a 1
    max_val = np.max(np.abs(estimate_np))
    if max_val > 0:
        estimate_np = estimate_np / max_val
    
    # display(Audio(estimate_np, rate=rate))

print("Fase 2 = Completada (Separación de la voz del audio)")
# ---------------------------------------------------

# Crear la carpeta si no existe
song_name = os.path.splitext(os.path.basename(audio_path))[0]
output_dir = os.path.join("Separado", song_name)
os.makedirs(output_dir, exist_ok=True)

# Guardar las voces separadas en un archivo
vocals = estimates['vocals'].detach().cpu().numpy()[0]
vocals_path = os.path.join(output_dir, "vocals.wav")
sf.write(vocals_path, vocals.T, rate)  # Guardar el archivo de audio

# Guardar el residual en un archivo
residual = estimates['residual'].detach().cpu().numpy()[0]
residual_path = os.path.join(output_dir, "residual.wav")
sf.write(residual_path, residual.T, rate)  # Guardar el archivo de audio

# Mostrar los archivos de audio guardados
# print(f"Archivos de audio guardados en: {output_dir}")
path_only_voice = vocals_path

print("Fase 3 = Completada (Guardado del archivo de la voz .wav)")
# ---------------------------------------------------

model = load_model("medium")
# model = load_model("base")

audio_path = path_only_voice
pred_out = transcribe(model, audio = audio_path, language="es")
# print(pred_out["text"])

print("Fase 4 = Completada (Transcipción de voz a texto)")
# ---------------------------------------------------

# Guardar el texto transcrito en un archivo
path_lyric_txt = os.path.join(output_dir, "Letra.txt")
transcribed_text = pred_out["text"]


with open(path_lyric_txt, 'w', encoding='utf-8') as file:
    file.write(transcribed_text)

# print(f'Texto transcrito guardado en {path_lyric_txt}')
print("Fase 5 = Completada (Guardado del archivo de texto .txt)")
# ---------------------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path_checkpoint = "Checkpoint/modelo_BETO_ultimo_checkpoint_state_dict.pth"

PRE_TRAINED_MODEL_NAME = 'ignacio-ave/BETO-nlp-sentiment-analysis-spanish'

category_mapping = {
    '0': 0,
    '1': 1,
}

class BETOMusicForKidsClassifier(nn.Module):

  def __init__(self, n_classes):
    super(BETOMusicForKidsClassifier, self).__init__()

    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
    outputs = self.bert(
                        input_ids = input_ids,
                        attention_mask = attention_mask
                        )
    pooled_output = outputs.pooler_output
    drop_output = self.drop(pooled_output)
    output = self.linear(drop_output)
    return output

# Inicializa el modelo
n_classes = len(category_mapping)
model = BETOMusicForKidsClassifier(n_classes)

# Cargar el state_dict
try:
    checkpoint = torch.load(path_checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    exit(1)

# Mueve el modelo al dispositivo adecuado
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

def predict_genre(text, model, tokenizer, device, max_length=200):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted_class = torch.max(outputs, dim=1)

    # Mapear el número de clase predicho al nombre de la categoría correspondiente
    predicted_genre = list(category_mapping.keys())[list(category_mapping.values()).index(predicted_class.item())]
    return predicted_genre

# Lee el contenido del archivo .txt
with open(path_lyric_txt, 'r', encoding='utf-8') as file:
    transcribed_text = file.read()

# Ajusta el texto transcrito para una impresión más legible
wrapped_lines = textwrap.wrap(transcribed_text, width=70)

# Imprime el contenido del archivo ajustado línea por línea
for line in wrapped_lines:
    print(line)

# Realiza la predicción de género utilizando el texto transcribido
genre = predict_genre(transcribed_text, model, tokenizer, device)
print(f"\n Predicción (0 = No Apta, 1 = Apta): {genre}")