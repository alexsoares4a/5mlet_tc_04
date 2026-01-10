# Usa uma imagem leve do Python
FROM python:3.11-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências
COPY requirements.txt .

# Instala as dependências (sem cache para reduzir tamanho)
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código do projeto para dentro do container
COPY . .

# Expõe a porta 8000 (onde a API vai rodar)
EXPOSE 8000

# Comando que será executado quando o container iniciar
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]