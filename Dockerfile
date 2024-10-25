FROM python:3.9-slim 
WORKDIR /app 

COPY requirements.txt requirements.txt 
COPY app.py app.py 
COPY lstm_model.keras lstm_model.keras
COPY tokenizer.pkl tookenizer.pkl

RUN pip install -r requirements.txt 

CMD ["streamlit", "run", "app.py"]
