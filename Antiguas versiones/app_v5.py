import os
import torch
import numpy as np
import stempeg
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from openunmix import predict
from torch import nn
from transformers import BertModel, BertTokenizer
from pydub import AudioSegment
import whisper
import soundfile as sf

app = Flask(__name__)
CORS(app)

# Configuración del modelo y el tokenizador
PRE_TRAINED_MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-uncased'
path_checkpoint = "Checkpoint/modelo_BETO_ultimo_checkpoint_state_dict_v2.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            input_ids=input_ids,
            attention_mask=attention_mask
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

    predicted_genre = list(category_mapping.keys())[list(category_mapping.values()).index(predicted_class.item())]
    return predicted_genre

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/js/<path:path>')
def send_js(path):
    return send_from_directory('static/js', path)

@app.route('/classify', methods=['POST'])
def classify():
    if 'audio' in request.files:
        audio_file = request.files['audio']
        audio_path = os.path.join("uploads", audio_file.filename)
        audio_file.save(audio_path)

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
        if (max_val > 0):
            audio_stems = audio_stems / max_val

        try:
            estimates = predict.separate(
                audio=torch.as_tensor(audio_stems).float(),
                rate=44100,
                targets=['vocals'],
                residual=True,
                device=device
            )
        except Exception as e:
            return jsonify({'error': f"Error during separation: {e}"}), 500

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

        # Transcribir el audio usando Whisper
        model_whisper = whisper.load_model("medium")
        pred_out = whisper.transcribe(model_whisper, audio=vocals_path, language="es")

        # Guardar el texto transcrito en un archivo
        path_lyric_txt = os.path.join(output_dir, "Letra.txt")
        transcribed_text = pred_out["text"]

        with open(path_lyric_txt, 'w', encoding='utf-8') as file:
            file.write(transcribed_text)

        # Realizar la predicción de género utilizando el texto transcribido
        genre = predict_genre(transcribed_text, model, tokenizer, device)
        
        return jsonify({
            'Predicción (0 = No Apta, 1 = Apta)': genre,
            'Transcription': transcribed_text
        })
    elif request.is_json:
        data = request.get_json()
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        genre = predict_genre(text, model, tokenizer, device)
        return jsonify({'Predicción (0 = No Apta, 1 = Apta)': genre})

    return jsonify({'error': 'Unsupported Media Type'}), 415

if __name__ == '__main__':
    app.run(debug=True)
