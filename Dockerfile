# Usando a imagem base do Python
FROM python:3.12-slim

# Definindo o diretório de trabalho dentro do container
WORKDIR /app

# Copiar os arquivos do projeto para dentro do container
COPY . /app

# Atualizar pip e instalar as dependências do sistema
RUN pip install --upgrade pip

# Instalar as dependências do seu projeto (assumindo que você tenha um arquivo requirements.txt)
RUN pip install -r requirements.txt

# Expõe a porta 8080 do container (mesma porta do seu código)
EXPOSE 8080

# Comando para rodar o servidor Flask
CMD ["python", "index.py"]
