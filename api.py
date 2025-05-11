from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import os
from data_validator import DataValidator
import uvicorn
from typing import Optional
import shutil
from datetime import datetime

app = FastAPI(
    title="Bulk Upload Validation API",
    description="API for validating and processing bulk uploads using AI-powered validation",
    version="1.0.0"
)

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("validated_data", exist_ok=True)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file for validation.
    
    Args:
        file: The file to be uploaded and validated
        
    Returns:
        JSON response with validation results
    """
    try:
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save uploaded file
        file_path = f"uploads/{timestamp}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize validator
        validator = DataValidator(model_name='sports_data')
        
        # Check if model exists, if not train it
        model_path = os.path.join("models", "sports_data_model.joblib")
        if not os.path.exists(model_path):
            # Train model on the uploaded data
            data = validator.load_data(file_path)
            validator.train_validation_model(data, save_path="models")
        
        # Validate the uploaded data
        corrected_data, report = validator.validate_and_correct(file_path, model_dir="models")
        
        # Save corrected data
        output_path = f"validated_data/{timestamp}_corrected_{file.filename}"
        corrected_data.to_csv(output_path, index=False)
        
        return JSONResponse({
            "message": "File processed successfully",
            "validation_report": report,
            "corrected_file_path": output_path
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/status")
async def get_model_status():
    """
    Check the status of the validation model.
    
    Returns:
        JSON response with model status
    """
    model_path = os.path.join("models", "sports_data_model.joblib")
    return {
        "model_exists": os.path.exists(model_path),
        "model_path": model_path if os.path.exists(model_path) else None
    }

@app.post("/model/train")
async def train_model(file: UploadFile = File(...)):
    """
    Train a new validation model using the provided data.
    
    Args:
        file: Training data file
        
    Returns:
        JSON response with training status
    """
    try:
        # Save uploaded file
        file_path = f"uploads/training_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Initialize and train validator
        validator = DataValidator(model_name='sports_data')
        data = validator.load_data(file_path)
        validator.train_validation_model(data, save_path="models")
        
        return {"message": "Model trained successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 