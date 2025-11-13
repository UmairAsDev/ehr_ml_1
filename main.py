import uvicorn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import router
from fastapi import FastAPI 




app = FastAPI(title="Psoriasis Flare Prediction API", version="1.0")
app.include_router(router)


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    
    
if __name__ == "__main__":
    main()