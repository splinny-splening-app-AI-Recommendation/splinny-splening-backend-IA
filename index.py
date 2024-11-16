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

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Diretório onde as imagens temporárias serão salvas
TEMP_DIR = 'temp_images/'

# Função para garantir que o diretório temporário exista
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

print(f'PyTorch version: {torch.__version__}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Caminho do modelo treinado
model_path = 'modelo_resnet18.pth'

# Definir a arquitetura do modelo com base na estrutura usada no treinamento
net = models.resnet18(pretrained=False)
num_features = net.fc.in_features
num_classes = 52  # Altere para o número real de classes no seu dataset

net.fc = nn.Linear(num_features, num_classes)
checkpoint = torch.load(model_path, map_location=device)
net.load_state_dict(checkpoint)
net = net.to(device)
net.eval()

# Lista de classes (substitua por sua lista completa de classes)
CLASSES = [
    'apple', 'avocado', 'bacon', 'bagels', 'banana', 'beans', 'beef', 'blackberries',
    'bread', 'broccoli', 'butter', 'cabbage', 'carrots', 'cauliflower', 'celery',
    'cheese', 'cherries', 'chicken', 'chocolate', 'coconut', 'corn', 'crab', 'cranberries',
    'cucumber', 'dates', 'eggs', 'fish', 'garlic', 'grapes', 'ham', 'honey', 'lemon',
    'lettuce', 'limes', 'mangos', 'milk', 'mushrooms', 'noise', 'onion', 'peppers',
    'potatoes', 'raddish', 'raspberries', 'rhubarb', 'rice', 'sausages', 'spinach',
    'sweetpotato', 'tofu', 'tomatoes', 'watermelon', 'yogurt'
]

# Rota para classificar imagens
@app.route('/classificar', methods=['POST'])
def classify_images():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhuma imagem foi enviada'}), 400

    files = request.files.getlist('file')
    if len(files) == 0:
        return jsonify({'error': 'Nenhuma imagem foi enviada'}), 400

    predictions = []
    for file in files:
        file_path = os.path.join(TEMP_DIR, file.filename)
        file.save(file_path)
        print(f"Imagem salva: {file_path}")

        file.seek(0)
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
            print(f"Imagem: {file.filename} - Classificação: {class_name}")

        except Exception as e:
            print(f"Erro ao processar a imagem {file.filename}: {str(e)}")
            predictions.append({
                'filename': file.filename,
                'error': f'Ocorreu um erro ao processar a imagem: {str(e)}'
            })

    return jsonify({'predictions': predictions})

# Rota para obter lista de URLs das imagens temporárias
@app.route('/temp_images', methods=['GET'])
def listar_imagens_temp():
    imagens = []
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        if os.path.isfile(file_path):
            imagens.append(f"{request.host_url}temp_images/{filename}")
    print("Lista de imagens:", imagens)
    return jsonify(imagens)

# Rota para acessar cada imagem individualmente
@app.route('/temp_images/<filename>', methods=['GET'])
def obter_imagem_temp(filename):
    return send_from_directory(TEMP_DIR, filename)

# Função para excluir as imagens temporárias ao encerrar o servidor
def delete_temp_images():
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        try:
            os.remove(file_path)
            print(f"Imagem {filename} excluída com sucesso.")
        except Exception as e:
            print(f"Erro ao excluir a imagem {filename}: {e}")

atexit.register(delete_temp_images)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
