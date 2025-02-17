from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
import io
import uvicorn
from pathlib import Path
import os

app = FastAPI(title="Background Remover API")

async def process_image(image_data: bytes, tolerance: int = 30) -> bytes:
    """Process image data and return processed image as bytes"""
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_data)).convert('RGBA')
        
        # Convert to numpy array
        data = np.array(img)
        
        # Get background color from top-left pixel
        bg_color = data[0, 0, :3]
        
        # Create alpha mask
        alpha = np.ones_like(data[:,:,3])
        
        # Make background transparent
        for i in range(3):
            alpha &= np.abs(data[:,:,i] - bg_color[i]) > tolerance
        
        # Apply alpha mask
        data[:,:,3] = alpha * 255
        
        # Create new image
        result = Image.fromarray(data)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/remove-background/")
async def remove_background(
    file: UploadFile = File(...),
    tolerance: int = 30
):
    """
    Remove background from uploaded image
    
    Parameters:
    - file: Image file (PNG or JPEG)
    - tolerance: Color matching tolerance (0-255, default: 30)
    
    Returns:
    - PNG image with transparent background
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read file
    contents = await file.read()
    
    # Process image
    result = await process_image(contents, tolerance)
    
    # Create temporary file for response
    temp_file = Path(f"/tmp/{file.filename}_processed.png")
    temp_file.write_bytes(result)
    
    return FileResponse(
        temp_file,
        media_type='image/png',
        filename=f"{file.filename}_nobg.png"
    )

@app.get("/")
async def root():
    """API root - provides basic information"""
    return {
        "name": "Background Remover API",
        "version": "1.0.0",
        "endpoints": {
            "POST /remove-background/": "Remove background from image",
            "GET /": "This information"
        }
    }

# For local development
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
