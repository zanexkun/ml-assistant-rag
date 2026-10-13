FROM python:3.11-slim
WORKDIR /rag 
COPY fetch_wiki.py build_index.py ask.py requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
RUN python fetch_wiki.py
RUN python build_index.py
ENTRYPOINT ["python","ask.py"]