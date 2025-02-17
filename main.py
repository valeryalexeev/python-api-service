from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
from pathlib import Path
import io
from datetime import datetime
from collections import deque
import uvicorn

app = FastAPI(title="Message Bubble Processor")

def color_distance(color1, color2):
    """Calculate Euclidean distance between two RGB colors"""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))

async def process_image(image_data: bytes, tolerance: float = 3.0) -> bytes:
    """Process message bubble image and return processed image as bytes"""
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_data)).convert('RGBA')
        width, height = img.size

        # Convert to numpy array
        data = np.array(img)

        # Get the background color (third pixel from top-left)
        background_color = data[0, 2, :3]  # RGB values of the third pixel

        # Create alpha mask (255 for visible, 0 for transparent)
        alpha = np.full((height, width), 255, dtype=np.uint8)

        # Flood fill queue
        queue = deque([(0, 0)])  # Start from top-left corner
        visited = set()

        # Flood fill to find background pixels
        while queue:
            y, x = queue.popleft()
            if (y, x) in visited:
                continue
                
            visited.add((y, x))
            current_color = data[y, x, :3]

            # If color is similar to background, mark as transparent
            if color_distance(current_color, background_color) <= tolerance:
                alpha[y, x] = 0
                # Add neighboring pixels
                for ny, nx in [(y+1, x), (y-1, x), (y, x+1), (y, x-1)]:
                    if (0 <= ny < height and 0 <= nx < width and
                        (ny, nx) not in visited):
                        queue.append((ny, nx))

        # Apply alpha mask
        data[:, :, 3] = alpha

        # Create new image
        result = Image.fromarray(data)

        # Save to bytes
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

@app.post("/remove-background/")
async def remove_background(
    file: UploadFile = File(...),
    tolerance: float = 3.0
):
    """
    Remove background from message bubble image using flood fill
    
    Parameters:
    - file: Image file (PNG or JPEG)
    - tolerance: Color difference tolerance (default: 3.0)
    
    Returns:
    - PNG image with transparent background
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        result = await process_image(contents, tolerance)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result-{timestamp}.png"
        
        # Create temporary file for response
        temp_file = Path(f"/tmp/{filename}")
        temp_file.write_bytes(result)
        
        return FileResponse(
            temp_file,
            media_type='image/png',
            filename=filename
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
