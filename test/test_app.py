"""
Test script to verify the Streamlit app can be imported
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    # Test import
    from sds_rag.ui import streamlit_app
    print("SUCCESS: Streamlit app imported successfully")
    
    # Test PDF extractor import
    from sds_rag.services.pdf_extracting_service import PDFExtractor
    print("SUCCESS: PDFExtractor imported successfully")
    
    print("Ready to run: poetry run streamlit run src/sds_rag/ui/streamlit_app.py")
    
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)