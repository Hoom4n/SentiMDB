FROM python:3.9-slim

WORKDIR /app

# Copy necessary files
COPY ./app ./app
COPY ./model ./model
COPY ./requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
   gcc \
    python3-dev \
   && rm -rf /var/lib/apt/lists/*


#Requirments
RUN pip install --no-cache-dir -r requirements.txt

#NLTK Data
RUN python -m nltk.downloader punkt wordnet punkt_tab

# Expose port and run the app
EXPOSE 5000

CMD ["python", "app/app.py"]