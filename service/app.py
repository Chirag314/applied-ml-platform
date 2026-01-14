from fastapi import FastAPI

app = FastAPI(title='Applied ML Platform API')

@app.get('/health')
def health():
    return {'status':'ok'}
