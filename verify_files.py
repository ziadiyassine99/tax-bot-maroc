
import sys
import os

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_module_config
from document_loader import DocumentProcessor

def check_for_text():
    module_id = "cgi"
    print(f"Checking module: {module_id}")
    
    try:
        config = get_module_config(module_id)
        source_path = config["pdf_path"]
        print(f"Source path: {source_path}")
        
        # Manually list files to see what python sees
        files = [f for f in os.listdir(source_path) if f.lower().endswith(('.pdf', '.docx', '.doc'))]
        print(f"Files in directory: {files}")
        
        if "CGI 2026.docx" in files:
            print("WARNING: CGI 2026.docx FOUND in directory.")
        else:
            print("NOTICE: CGI 2026.docx NOT in directory.")
            
        if "CGI 2026.doc" in files:
            print("NOTICE: CGI 2026.doc FOUND in directory.")
        
        processor = DocumentProcessor(pdf_path=source_path)
        # We will hook into the internal method to confirm what files are processed if possible, 
        # or just trust the load.
        
        print("Loading documents...")
        documents = processor.load_documents()
        print(f"Loaded {len(documents)} document pages/sections.")
        
        target = "37,75"
        found = False
        
        for doc in documents:
            if target in doc.page_content:
                source = doc.metadata.get("source", "Unknown")
                print(f"FOUND in: {source}")
                found = True
                
        if not found:
            print("Text NOT found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_for_text()
