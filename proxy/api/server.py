import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import requests

app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/proxy")
async def proxy(request: Request):
    try:
        # Get the JSON payload from the request
        payload = await request.json()
        
        print(payload)
        
        # Extract the target URL and request type from the payload
        target_url = payload.get("target_url")
        request_type = payload.get("request_type", "POST").upper()  # Default to POST if not specified
        
        if not target_url:
            raise HTTPException(status_code=400, detail="Target URL is required")
        
        if request_type not in ["GET", "POST"]:
            raise HTTPException(status_code=400, detail="Invalid request type. Use 'GET' or 'POST'.")
        
        # Remove the target_url and request_type from the payload before forwarding
        payload_to_forward = payload.copy()
        payload_to_forward.pop("target_url", None)
        payload_to_forward.pop("request_type", None)

        print('jo')
        
        print(target_url, payload_to_forward)
        # Forward the request to the target URL
        if request_type == "GET":
            response = requests.get(target_url, params=payload_to_forward)
        else:  # POST
            response = requests.post(target_url, json=payload_to_forward)

        # Return the response from the target URL
        return response.json()

 
        return response.json()
    
    except httpx.RequestError as exc:
        raise HTTPException(status_code=500, detail=f"An error occurred while requesting {exc.request.url!r}.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)