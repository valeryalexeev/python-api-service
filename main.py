from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
from pathlib import Path
import io
import uvicorn

app = FastAPI(title="Message Bubble Processor")

async def process_image(image_data: bytes) -> bytes:
    """Process message bubble image and return processed image as bytes"""
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_data)).convert('RGBA')
        
        # Convert to numpy array
        data = np.array(img)
        
        # Calculate brightness (using standard coefficients for grayscale conversion)
        brightness = np.sum(data[:,:,:3] * [0.299, 0.587, 0.114], axis=2)
        
        # Create alpha mask - keep darker pixels (text)
        alpha = (brightness < 200).astype(np.uint8) * 255
        
        # Remove noise/gray areas
        alpha[brightness > 150] = 0
        
        # Apply alpha mask
        data[:,:,3] = alpha
        
        # Create new image
        result = Image.fromarray(data)
        
        # Save to bytes
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
        
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

@app.post("/remove-background/")
async def remove_background(file: UploadFile = File(...)):
    """
    Remove background from message bubble image
    
    Parameters:
    - file: Image file (PNG or JPEG)
    
    Returns:
    - PNG image with transparent background
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        result = await process_image(contents)
        
        # Create temporary file for response
        temp_file = Path(f"/tmp/{file.filename}_processed.png")
        temp_file.write_bytes(result)
        
        return FileResponse(
            temp_file,
            media_type='image/png',
            filename=f"{file.filename}_nobg.png"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
