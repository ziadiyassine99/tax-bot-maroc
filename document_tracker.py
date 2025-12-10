"""
Document tracker module for incremental indexing.
Tracks file hashes to detect changes and enable efficient re-indexing.
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict


TRACKER_FILE = ".document_tracker.json"


@dataclass
class DocumentInfo:
    """Information about a tracked document."""
    file_path: str
    file_hash: str
    last_indexed: str
    chunk_count: int
    file_size: int


class DocumentTracker:
    """
    Tracks document files and their hashes for incremental indexing.
    Detects added, modified, and deleted files.
    """
    
    def __init__(self, tracker_path: str = TRACKER_FILE):
        """
        Initialize the document tracker.
        
        Args:
            tracker_path: Path to the JSON file storing tracking data
        """
        self.tracker_path = tracker_path
        self._data: Dict[str, Dict[str, DocumentInfo]] = {}
        self._load()
    
    def _load(self) -> None:
        """Load tracking data from disk."""
        if os.path.exists(self.tracker_path):
            try:
                with open(self.tracker_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                    # Convert raw dicts back to DocumentInfo objects
                    for collection, docs in raw_data.items():
                        self._data[collection] = {
                            path: DocumentInfo(**info) 
                            for path, info in docs.items()
                        }
            except (json.JSONDecodeError, TypeError):
                self._data = {}
        else:
            self._data = {}
    
    def _save(self) -> None:
        """Save tracking data to disk."""
        # Convert DocumentInfo objects to dicts for JSON serialization
        raw_data = {
            collection: {
                path: asdict(info) 
                for path, info in docs.items()
            }
            for collection, docs in self._data.items()
        }
        with open(self.tracker_path, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """
        Compute MD5 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MD5 hash as hex string
        """
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def get_collection_docs(self, collection_name: str) -> Dict[str, DocumentInfo]:
        """Get all tracked documents for a collection."""
        return self._data.get(collection_name, {})
    
    def detect_changes(
        self, 
        collection_name: str, 
        current_files: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Detect which files have been added, modified, or deleted.
        
        Args:
            collection_name: Name of the collection/module
            current_files: List of current file paths
            
        Returns:
            Tuple of (added_files, modified_files, deleted_files)
        """
        tracked = self.get_collection_docs(collection_name)
        tracked_paths = set(tracked.keys())
        current_paths = set(current_files)
        
        # New files not in tracker
        added = list(current_paths - tracked_paths)
        
        # Files removed from directory
        deleted = list(tracked_paths - current_paths)
        
        # Files that exist but may have changed
        modified = []
        for path in current_paths & tracked_paths:
            if os.path.exists(path):
                current_hash = self.compute_file_hash(path)
                if current_hash != tracked[path].file_hash:
                    modified.append(path)
        
        return added, modified, deleted
    
    def update_document(
        self, 
        collection_name: str, 
        file_path: str, 
        chunk_count: int
    ) -> None:
        """
        Update tracking info for a document after indexing.
        
        Args:
            collection_name: Name of the collection/module
            file_path: Path to the document file
            chunk_count: Number of chunks created from this document
        """
        if collection_name not in self._data:
            self._data[collection_name] = {}
        
        self._data[collection_name][file_path] = DocumentInfo(
            file_path=file_path,
            file_hash=self.compute_file_hash(file_path),
            last_indexed=datetime.now().isoformat(),
            chunk_count=chunk_count,
            file_size=os.path.getsize(file_path)
        )
        self._save()
    
    def remove_document(self, collection_name: str, file_path: str) -> None:
        """
        Remove a document from tracking.
        
        Args:
            collection_name: Name of the collection/module
            file_path: Path to the document file
        """
        if collection_name in self._data and file_path in self._data[collection_name]:
            del self._data[collection_name][file_path]
            self._save()
    
    def clear_collection(self, collection_name: str) -> None:
        """Clear all tracking data for a collection."""
        if collection_name in self._data:
            del self._data[collection_name]
            self._save()
    
    def get_stats(self, collection_name: str) -> Dict:
        """Get statistics for a collection."""
        docs = self.get_collection_docs(collection_name)
        if not docs:
            return {"tracked_files": 0, "total_chunks": 0}
        
        return {
            "tracked_files": len(docs),
            "total_chunks": sum(d.chunk_count for d in docs.values()),
            "total_size_mb": round(sum(d.file_size for d in docs.values()) / (1024 * 1024), 2),
            "files": list(docs.keys())
        }


def get_pdf_files_in_directory(directory: str) -> List[str]:
    """
    Get all PDF files in a directory (recursive).
    
    Args:
        directory: Path to search for PDFs
        
    Returns:
        List of PDF file paths
    """
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

