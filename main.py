from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
from sqlalchemy import create_engine, Column, Integer, String, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Optional

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

# Database Setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./recipes.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Model
class RecipeDB(Base):
    __tablename__ = "recipes"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    ingredients = Column(JSON)
    instructions = Column(JSON)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(title=APP_NAME)

# Mount static files directory
app.mount("/static", StaticFiles(directory="."), name="static")

# Pydantic Models
class RecipeText(BaseModel):
    text: str

class RecipeCreate(BaseModel):
    title: str
    ingredients: List[str]
    instructions: List[str]

class RecipeResponse(BaseModel):
    id: int
    title: str
    ingredients: List[str]
    instructions: List[str]
    
    class Config:
        from_attributes = True

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

@app.post("/api/recipes", response_model=RecipeResponse)
async def create_recipe(recipe: RecipeCreate, db: Session = Depends(get_db)):
    """POST endpoint to create a new recipe in the database"""
    try:
        db_recipe = RecipeDB(
            title=recipe.title,
            ingredients=recipe.ingredients,
            instructions=recipe.instructions
        )
        db.add(db_recipe)
        db.commit()
        db.refresh(db_recipe)
        return RecipeResponse(
            id=db_recipe.id,
            title=db_recipe.title,
            ingredients=db_recipe.ingredients,
            instructions=db_recipe.instructions
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating recipe: {str(e)}")

@app.get("/api/recipes", response_model=List[RecipeResponse])
async def get_recipes(db: Session = Depends(get_db)):
    """GET endpoint to retrieve all recipes from the database"""
    try:
        recipes = db.query(RecipeDB).all()
        return [
            RecipeResponse(
                id=recipe.id,
                title=recipe.title,
                ingredients=recipe.ingredients,
                instructions=recipe.instructions
            )
            for recipe in recipes
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving recipes: {str(e)}")

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
