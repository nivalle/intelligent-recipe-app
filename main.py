from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Intelligent Recipe App")

# Mount static files directory
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def read_index():
    """Serve the index.html file at the root URL"""
    return FileResponse("index.html")

@app.get("/api/hello")
async def hello():
    """GET endpoint that returns a JSON response"""
    return {"message": "Hello, Recipe App!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
