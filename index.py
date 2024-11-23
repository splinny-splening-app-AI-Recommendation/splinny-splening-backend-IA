import os
import atexit
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torch.nn as nn
from torchvision import models
import requests

# Configuração do Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Diretório onde as imagens temporárias serão salvas
TEMP_DIR = 'temp_images/'

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Função para pré-processar a imagem com padding centralizado e redimensionamento
def transform_image_with_padding(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    width, height = image.size

    padding_left = (224 - width) // 2
    padding_top = (224 - height) // 2
    padding_right = 224 - width - padding_left
    padding_bottom = 224 - height - padding_top

    transform = transforms.Compose([
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=(0, 0, 0)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Configuração do PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregamento do modelo
model_path = 'modelo_resnet18.pth'
net = models.resnet18(pretrained=False)
num_features = net.fc.in_features
num_classes = 52  # Número de classes do dataset
net.fc = nn.Linear(num_features, num_classes)

checkpoint = torch.load(model_path, map_location=device)
net.load_state_dict(checkpoint)
net = net.to(device)
net.eval()

# Lista de classes
CLASSES = [
    'apple', 'avocado', 'bacon', 'bagels', 'banana', 'beans', 'beef', 'blackberries',
    'bread', 'broccoli', 'butter', 'cabbage', 'carrots', 'cauliflower', 'celery',
    'cheese', 'cherries', 'chicken', 'chocolate', 'coconut', 'corn', 'crab', 'cranberries',
    'cucumber', 'dates', 'eggs', 'fish', 'garlic', 'grapes', 'ham', 'honey', 'lemon',
    'lettuce', 'limes', 'mangos', 'milk', 'mushrooms', 'noise', 'onion', 'peppers',
    'potatoes', 'raddish', 'raspberries', 'rhubarb', 'rice', 'sausages', 'spinach',
    'sweetpotato', 'tofu', 'tomatoes', 'watermelon', 'yogurt'
]

# URL do backend de recomendação
RECOMMENDATION_BACKEND_URL = "http://127.0.0.1:8082/recommend"  # Altere se necessário

@app.route('/classificar', methods=['POST'])
def classify_and_send():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhuma imagem foi enviada'}), 400

    files = request.files.getlist('file')
    if len(files) == 0:
        return jsonify({'error': 'Nenhuma imagem foi enviada'}), 400

    predictions = []
    for file in files:
        try:
            img_bytes = file.read()
            img_tensor = transform_image_with_padding(img_bytes).to(device)

            with torch.no_grad():
                outputs = net(img_tensor)
                _, predicted = torch.max(outputs.data, 1)

            class_name = CLASSES[predicted.item()]
            predictions.append({
                'filename': file.filename,
                'classificacao': class_name
            })
        except Exception as e:
            predictions.append({
                'filename': file.filename,
                'error': f'Ocorreu um erro ao processar a imagem: {str(e)}'
            })

    # Enviar JSON para o backend de recomendação
    try:
        response = requests.post(RECOMMENDATION_BACKEND_URL, json={'ingredients': predictions})
        if response.status_code == 200:
            return jsonify({'status': 'success', 'recommendations': response.json()})
        else:
            return jsonify({'error': f'Falha ao enviar para o backend de recomendação: {response.status_code}'}), 500
    except Exception as e:
        return jsonify({'error': f'Ocorreu um erro ao conectar ao backend de recomendação: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
