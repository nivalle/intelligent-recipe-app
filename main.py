from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime, Float, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from typing import List, Optional
from datetime import datetime

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

# SQLAlchemy Models
class RecipeDB(Base):
    __tablename__ = "recipes"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    ingredients = Column(JSON)
    instructions = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    parent_id = Column(Integer, ForeignKey("recipes.id"), nullable=True)
    
    # Relationships
    versions = relationship("RecipeDB", backref="parent", remote_side=[id])
    photos = relationship("RecipePhoto", back_populates="recipe")
    ratings = relationship("RecipeRating", back_populates="recipe")

class RecipePhoto(Base):
    __tablename__ = "recipe_photos"
    
    id = Column(Integer, primary_key=True, index=True)
    recipe_id = Column(Integer, ForeignKey("recipes.id"))
    photo_data = Column(Text)  # Base64 encoded image
    photo_description = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    recipe = relationship("RecipeDB", back_populates="photos")

class RecipeRating(Base):
    __tablename__ = "recipe_ratings"
    
    id = Column(Integer, primary_key=True, index=True)
    recipe_id = Column(Integer, ForeignKey("recipes.id"))
    rating = Column(Float)  # 1-5 stars
    review = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    recipe = relationship("RecipeDB", back_populates="ratings")

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
    parent_id: Optional[int] = None

class RecipeResponse(BaseModel):
    id: int
    title: str
    ingredients: List[str]
    instructions: List[str]
    created_at: datetime
    parent_id: Optional[int]
    
    class Config:
        from_attributes = True

class RecipePhotoCreate(BaseModel):
    photo_data: str  # Base64 encoded image
    photo_description: Optional[str] = None

class RecipePhotoResponse(BaseModel):
    id: int
    recipe_id: int
    photo_data: str
    photo_description: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

class RecipeRatingCreate(BaseModel):
    rating: float  # 1-5 stars
    review: Optional[str] = None

class RecipeRatingResponse(BaseModel):
    id: int
    recipe_id: int
    rating: float
    review: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True

class RecipeWithHistoryResponse(BaseModel):
    id: int
    title: str
    ingredients: List[str]
    instructions: List[str]
    created_at: datetime
    parent_id: Optional[int]
    versions: List['RecipeResponse'] = []
    photos: List[RecipePhotoResponse] = []
    ratings: List[RecipeRatingResponse] = []
    
    class Config:
        from_attributes = True

class ModificationRequest(BaseModel):
    prompt: str

class TransientRecipeModificationRequest(BaseModel):
    prompt: str
    ingredients: List[str]
    instructions: List[str]

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
            instructions=recipe.instructions,
            parent_id=recipe.parent_id
        )
        db.add(db_recipe)
        db.commit()
        db.refresh(db_recipe)
        return RecipeResponse(
            id=db_recipe.id,
            title=db_recipe.title,
            ingredients=db_recipe.ingredients,
            instructions=db_recipe.instructions,
            created_at=db_recipe.created_at,
            parent_id=db_recipe.parent_id
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
                instructions=recipe.instructions,
                created_at=recipe.created_at,
                parent_id=recipe.parent_id
            )
            for recipe in recipes
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving recipes: {str(e)}")

@app.get("/api/recipes/{recipe_id}", response_model=RecipeResponse)
async def get_recipe(recipe_id: int, db: Session = Depends(get_db)):
    """GET endpoint to retrieve a single recipe by ID"""
    try:
        recipe = db.query(RecipeDB).filter(RecipeDB.id == recipe_id).first()
        
        if recipe is None:
            raise HTTPException(status_code=404, detail=f"Recipe with ID {recipe_id} not found")
        
        return RecipeResponse(
            id=recipe.id,
            title=recipe.title,
            ingredients=recipe.ingredients,
            instructions=recipe.instructions,
            created_at=recipe.created_at,
            parent_id=recipe.parent_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving recipe: {str(e)}")

@app.post("/api/modify-transient-recipe")
async def modify_transient_recipe(modification_request: TransientRecipeModificationRequest):
    """POST endpoint to modify a transient recipe using AI"""
    
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Google API key not configured")
    
    if not model:
        raise HTTPException(status_code=500, detail="Gemini model not available")
    
    try:
        # Create a detailed prompt for recipe modification
        prompt = f"""
        Please modify the following recipe based on the user's request.
        
        Original Recipe:
        Ingredients: {modification_request.ingredients}
        Instructions: {modification_request.instructions}
        
        User's Modification Request: {modification_request.prompt}
        
        Please modify the recipe according to the user's request. Return ONLY a valid JSON object with this exact structure:
        {{
            "ingredients": ["modified ingredient 1", "modified ingredient 2", ...],
            "instructions": ["modified step 1", "modified step 2", ...]
        }}
        
        Rules:
        - Modify ingredients and instructions according to the user's request
        - Keep the same cooking method and general approach
        - Ensure the modified recipe is still functional and complete
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
            
            modified_recipe = json.loads(response_text)
            
            # Validate the structure
            if "ingredients" not in modified_recipe or "instructions" not in modified_recipe:
                raise ValueError("Missing required fields")
            
            if not isinstance(modified_recipe["ingredients"], list) or not isinstance(modified_recipe["instructions"], list):
                raise ValueError("Invalid field types")
            
            return modified_recipe
            
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse AI response as JSON: {str(e)}")
        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"Invalid response structure: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error modifying transient recipe: {str(e)}")

@app.post("/api/recipes/{recipe_id}/modify")
async def modify_recipe(recipe_id: int, modification_request: ModificationRequest, db: Session = Depends(get_db)):
    """POST endpoint to modify a recipe using AI"""
    
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Google API key not configured")
    
    if not model:
        raise HTTPException(status_code=500, detail="Gemini model not available")
    
    try:
        # Get the original recipe from the database
        recipe = db.query(RecipeDB).filter(RecipeDB.id == recipe_id).first()
        
        if recipe is None:
            raise HTTPException(status_code=404, detail=f"Recipe with ID {recipe_id} not found")
        
        # Create a detailed prompt for recipe modification
        prompt = f"""
        Please modify the following recipe based on the user's request.
        
        Original Recipe:
        Title: {recipe.title}
        Ingredients: {recipe.ingredients}
        Instructions: {recipe.instructions}
        
        User's Modification Request: {modification_request.prompt}
        
        Please modify the recipe according to the user's request. Return ONLY a valid JSON object with this exact structure:
        {{
            "ingredients": ["modified ingredient 1", "modified ingredient 2", ...],
            "instructions": ["modified step 1", "modified step 2", ...]
        }}
        
        Rules:
        - Modify ingredients and instructions according to the user's request
        - Keep the same cooking method and general approach
        - Ensure the modified recipe is still functional and complete
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
            
            modified_recipe = json.loads(response_text)
            
            # Validate the structure
            if "ingredients" not in modified_recipe or "instructions" not in modified_recipe:
                raise ValueError("Missing required fields")
            
            if not isinstance(modified_recipe["ingredients"], list) or not isinstance(modified_recipe["instructions"], list):
                raise ValueError("Invalid field types")
            
            return modified_recipe
            
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse AI response as JSON: {str(e)}")
        except ValueError as e:
            raise HTTPException(status_code=500, detail=f"Invalid response structure: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error modifying recipe: {str(e)}")

@app.post("/api/recipes/{recipe_id}/save-modified")
async def save_modified_recipe(recipe_id: int, recipe_data: RecipeCreate, db: Session = Depends(get_db)):
    """POST endpoint to save a modified version of a recipe"""
    try:
        # Create new recipe version
        new_recipe = RecipeDB(
            title=recipe_data.title,
            ingredients=recipe_data.ingredients,
            instructions=recipe_data.instructions,
            parent_id=recipe_id
        )
        
        db.add(new_recipe)
        db.commit()
        db.refresh(new_recipe)
        
        return RecipeResponse.from_orm(new_recipe)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving modified recipe: {str(e)}")

@app.get("/api/recipes/{recipe_id}/history")
async def get_recipe_history(recipe_id: int, db: Session = Depends(get_db)):
    """GET endpoint to retrieve recipe history including all versions"""
    try:
        # Get the base recipe
        base_recipe = db.query(RecipeDB).filter(RecipeDB.id == recipe_id).first()
        
        if base_recipe is None:
            raise HTTPException(status_code=404, detail=f"Recipe with ID {recipe_id} not found")
        
        # Get all versions (including the base recipe)
        all_versions = []
        
        # If this is a version, find the base recipe
        if base_recipe.parent_id:
            base_recipe = db.query(RecipeDB).filter(RecipeDB.id == base_recipe.parent_id).first()
            if base_recipe is None:
                raise HTTPException(status_code=404, detail="Base recipe not found")
        
        # Add base recipe
        all_versions.append(RecipeResponse.from_orm(base_recipe))
        
        # Add all versions
        versions = db.query(RecipeDB).filter(RecipeDB.parent_id == base_recipe.id).order_by(RecipeDB.created_at).all()
        for version in versions:
            all_versions.append(RecipeResponse.from_orm(version))
        
        return {
            "base_recipe": RecipeResponse.from_orm(base_recipe),
            "versions": all_versions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving recipe history: {str(e)}")

@app.post("/api/recipes/{recipe_id}/photos")
async def add_recipe_photo(recipe_id: int, photo_data: RecipePhotoCreate, db: Session = Depends(get_db)):
    """POST endpoint to add a photo to a recipe"""
    try:
        # Check if recipe exists
        recipe = db.query(RecipeDB).filter(RecipeDB.id == recipe_id).first()
        if recipe is None:
            raise HTTPException(status_code=404, detail=f"Recipe with ID {recipe_id} not found")
        
        # Create new photo
        new_photo = RecipePhoto(
            recipe_id=recipe_id,
            photo_data=photo_data.photo_data,
            photo_description=photo_data.photo_description
        )
        
        db.add(new_photo)
        db.commit()
        db.refresh(new_photo)
        
        return RecipePhotoResponse.from_orm(new_photo)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding photo: {str(e)}")

@app.get("/api/recipes/{recipe_id}/photos")
async def get_recipe_photos(recipe_id: int, db: Session = Depends(get_db)):
    """GET endpoint to retrieve all photos for a recipe"""
    try:
        # Check if recipe exists
        recipe = db.query(RecipeDB).filter(RecipeDB.id == recipe_id).first()
        if recipe is None:
            raise HTTPException(status_code=404, detail=f"Recipe with ID {recipe_id} not found")
        
        # Get all photos
        photos = db.query(RecipePhoto).filter(RecipePhoto.recipe_id == recipe_id).order_by(RecipePhoto.created_at.desc()).all()
        
        return [RecipePhotoResponse.from_orm(photo) for photo in photos]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving photos: {str(e)}")

@app.post("/api/recipes/{recipe_id}/ratings")
async def add_recipe_rating(recipe_id: int, rating_data: RecipeRatingCreate, db: Session = Depends(get_db)):
    """POST endpoint to add a rating to a recipe"""
    try:
        # Check if recipe exists
        recipe = db.query(RecipeDB).filter(RecipeDB.id == recipe_id).first()
        if recipe is None:
            raise HTTPException(status_code=404, detail=f"Recipe with ID {recipe_id} not found")
        
        # Validate rating
        if rating_data.rating < 1 or rating_data.rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        # Create new rating
        new_rating = RecipeRating(
            recipe_id=recipe_id,
            rating=rating_data.rating,
            review=rating_data.review
        )
        
        db.add(new_rating)
        db.commit()
        db.refresh(new_rating)
        
        return RecipeRatingResponse.from_orm(new_rating)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding rating: {str(e)}")

@app.get("/api/recipes/{recipe_id}/ratings")
async def get_recipe_ratings(recipe_id: int, db: Session = Depends(get_db)):
    """GET endpoint to retrieve all ratings for a recipe"""
    try:
        # Check if recipe exists
        recipe = db.query(RecipeDB).filter(RecipeDB.id == recipe_id).first()
        if recipe is None:
            raise HTTPException(status_code=404, detail=f"Recipe with ID {recipe_id} not found")
        
        # Get all ratings
        ratings = db.query(RecipeRating).filter(RecipeRating.recipe_id == recipe_id).order_by(RecipeRating.created_at.desc()).all()
        
        return [RecipeRatingResponse.from_orm(rating) for rating in ratings]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving ratings: {str(e)}")

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
