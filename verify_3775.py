
import sys
import os

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_module_config
from document_loader import DocumentProcessor

def check_for_text():
    # Target the CGI module as implied by the screenshot "CGI 2026.docx"
    module_id = "cgi"
    print(f"Checking module: {module_id}")
    
    try:
        config = get_module_config(module_id)
        source_path = config["pdf_path"]
        print(f"Source path: {source_path}")
        
        processor = DocumentProcessor(pdf_path=source_path)
        documents = processor.load_documents()
        
        print(f"Loaded {len(documents)} document pages/sections.")
        
        found = False
        target = "37,75"
        
        for i, doc in enumerate(documents):
            content = doc.page_content
            if target in content:
                found = True
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                print(f"\n✅ FOUND '{target}' in:")
                print(f"   - File: {source}")
                print(f"   - Page/Section: {page}")
                
                # Print context around the match
                idx = content.find(target)
                start = max(0, idx - 100)
                end = min(len(content), idx + 100)
                print(f"   - Context:\n...{content[start:end]}...\n")
                
        if not found:
            print(f"\n❌ Text '{target}' NOT FOUND in any loaded documents.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_for_text()
