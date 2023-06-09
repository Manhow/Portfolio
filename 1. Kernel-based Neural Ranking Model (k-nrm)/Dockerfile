# Start from a Python 3 base image
FROM python:3.9-slim-buster

# Set a directory for the application
WORKDIR /app

# Include the necessary files in the Docker image
COPY . /app

# Install the necessary libraries
RUN pip install flask numpy faiss-cpu nltk langdetect faiss-cpu torch torchvision torchaudio

# Expose a port for the Flask application
EXPOSE 5000

# Define an environment variable for the Flask application
ENV FLASK_APP=flask_app.py

# Define the environment variables for the paths
ENV EMB_PATH_KNRM=/app/data/knrm_emb.bin
ENV VOCAB_PATH=/app/data/vocab.json
ENV EMB_PATH_GLOVE=/app/glove.6B.50d.txt
ENV MLP_PATH=/app/data/knrm_mlp.bin

# Run the application
CMD ["flask", "run", "--host=0.0.0.0"]