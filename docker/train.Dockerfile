FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install -U pip && pip install -e .
CMD ['python','-c','print("train placeholder")']
