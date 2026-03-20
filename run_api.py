"""Entry point to run the CV Midterm API server.

Usage:
    py -3.10 run_api.py
    Then open http://localhost:8000/docs for Swagger UI.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
