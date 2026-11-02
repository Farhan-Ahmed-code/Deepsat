"""
DeepSatVision Application Runner
================================
Runs the unified FastAPI inference engine and serves the production React research website.

Usage:
    python run_app.py
    # Or:
    uvicorn backend.server:app --host 127.0.0.1 --port 8000
"""

import os
import sys
import webbrowser
import uvicorn

def main():
    print("=" * 70)
    print("   DEEPSATVISION: AI-BASED LAND USE CLASSIFICATION RESEARCH PLATFORM")
    print("   Amity University Uttar Pradesh, Noida - Minor Project (ETMN100)")
    print("=" * 70)
    print("-> Student Researchers: Farhan Ahmed Siddique & Adesh Pal")
    print("-> Faculty Guide:       Dr Aakanshi Gupta (Asst. Professor-III, Dept of CSE)")
    print("-> Programme:           B.Tech Computer Science & Engineering (Data Science)")
    print("=" * 70)
    print("[+] Launching unified FastAPI server & compiled React frontend on http://127.0.0.1:8000")
    print("    - Web Dashboard:     http://127.0.0.1:8000")
    print("    - API Documentation: http://127.0.0.1:8000/docs")
    print("    - Health Check:      http://127.0.0.1:8000/api/health")
    print("=" * 70)
    
    # Start server
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()

