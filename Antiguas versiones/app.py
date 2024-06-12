from flask import Flask, request, jsonify
import torch
from torch import nn
from transformers import BertModel, BertTokenizer, logging as transformers_logging
import os
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

# Configuración del modelo y el tokenizador
PRE_TRAINED_MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-uncased'
# PRE_TRAINED_MODEL_NAME = 'ignacio-ave/BETO-nlp-sentiment-analysis-spanish'
path_checkpoint = "Checkpoint/modelo_BETO_ultimo_checkpoint_state_dict_v3.pth"
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
transformers_logging.set_verbosity_error()
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

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    genre = predict_genre(text, model, tokenizer, device)
    return jsonify({'Predicción (0 = No Apta, 1 = Apta)': genre})

if __name__ == '__main__':
    app.run(debug=True)
