#!/usr/bin/env python3
"""
Launcher script for the PDF Extraction Streamlit app
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Get the path to the Streamlit app
    app_path = Path(__file__).parent / "src" / "sds_rag" / "ui" / "streamlit_app.py"
    
    # Run the Streamlit app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down the application...")
    except Exception as e:
        print(f"❌ Error running the application: {e}")

if __name__ == "__main__":
    main()