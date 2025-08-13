from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
APP_NAME = os.getenv("APP_NAME", "Intelligent Recipe App")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI(title=APP_NAME)

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

@app.get("/api/config")
async def get_config():
    """GET endpoint to check configuration (without exposing sensitive data)"""
    return {
        "app_name": APP_NAME,
        "debug": DEBUG,
        "host": HOST,
        "port": PORT,
        "gemini_api_configured": bool(GEMINI_API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting {APP_NAME} on {HOST}:{PORT}")
    if GEMINI_API_KEY:
        print("✅ Gemini API key loaded successfully")
    else:
        print("⚠️  Warning: Gemini API key not found in environment variables")
    
    if DEBUG:
        uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
    else:
        uvicorn.run(app, host=HOST, port=PORT)
