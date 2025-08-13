from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Load environment variables from .env file
load_dotenv()

# Get environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
APP_NAME = os.getenv("APP_NAME", "Intelligent Recipe App")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Configure Google Generative AI
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
else:
    model = None

app = FastAPI(title=APP_NAME)

# Mount static files directory
app.mount("/static", StaticFiles(directory="."), name="static")

# Pydantic model for request validation
class RecipeText(BaseModel):
    text: str

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
        "google_api_configured": bool(GOOGLE_API_KEY)
    }

@app.post("/api/parse-recipe")
async def parse_recipe(recipe_data: RecipeText):
    """POST endpoint to parse recipe text and extract ingredients and instructions"""
    
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Google API key not configured")
    
    if not model:
        raise HTTPException(status_code=500, detail="Gemini model not available")
    
    try:
        # Create the prompt for recipe parsing
        prompt = f"""
        Please analyze the following recipe text and extract the ingredients and instructions.
        
        Recipe text:
        {recipe_data.text}
        
        Return ONLY a valid JSON object with this exact structure:
        {{
            "ingredients": ["ingredient 1", "ingredient 2", ...],
            "instructions": ["step 1", "step 2", ...]
        }}
        
        Rules:
        - Ingredients should be extracted as individual strings in an array
        - Instructions should be extracted as numbered steps in an array
        - Return ONLY the JSON object, no additional text
        - Ensure the JSON is valid and properly formatted
        """
        
        # Generate response from Gemini
        response = model.generate_content(prompt)
        
        # Extract the text response
        response_text = response.text.strip()
        
        # Try to parse the JSON response
        try:
            # Remove any markdown formatting if present
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            parsed_recipe = json.loads(response_text)
            
            # Validate the structure
            if "ingredients" not in parsed_recipe or "instructions" not in parsed_recipe:
                raise ValueError("Missing required fields")
            
            if not isinstance(parsed_recipe["ingredients"], list) or not isinstance(parsed_recipe["instructions"], list):
                raise ValueError("Invalid field types")
            
            return parsed_recipe
            
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse AI response as JSON: {str(e)}")
        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"Invalid response structure: {str(e)}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing recipe: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 Starting {APP_NAME} on {HOST}:{PORT}")
    if GOOGLE_API_KEY:
        print("✅ Google API key loaded successfully")
    else:
        print("⚠️  Warning: Google API key not found in environment variables")
    
    if DEBUG:
        uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
    else:
        uvicorn.run(app, host=HOST, port=PORT)
