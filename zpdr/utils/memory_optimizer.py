#!/usr/bin/env python3
"""
Memory optimization utilities for ZPDR.

This module provides memory optimization capabilities for ZPDR operations,
reducing memory usage for large files and computationally intensive tasks.
"""

import os
import io
import mmap
import tempfile
from typing import List, Tuple, Dict, Any, Union, Optional, BinaryIO
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryOptimizer:
    """
    Utility for memory optimization in ZPDR operations.
    """
    
    def __init__(self, 
                 max_memory_mb: int = 100,
                 use_mmap: bool = True,
                 temp_dir: Optional[str] = None):
        """
        Initialize the memory optimizer.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            use_mmap: Whether to use memory-mapped files for large data
            temp_dir: Directory for temporary files
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.use_mmap = use_mmap
        self.temp_dir = temp_dir
        self.temp_files = []
        
        logger.info(f"Initialized MemoryOptimizer with {max_memory_mb}MB limit")
    
    def __del__(self):
        """
        Clean up temporary files on deletion.
        """
        self.cleanup()
    
    def cleanup(self):
        """
        Clean up temporary files.
        """
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Error cleaning up temporary file {temp_file}: {e}")
        
        self.temp_files = []
    
    def read_file_optimized(self, 
                           file_path: Union[str, Path], 
                           chunk_size: Optional[int] = None) -> Union[bytes, mmap.mmap]:
        """
        Read a file with memory optimization.
        
        For small files, reads the entire file into memory.
        For large files, uses memory mapping or streaming.
        
        Args:
            file_path: Path to the file to read
            chunk_size: Size of chunks for streaming (ignored if using mmap)
            
        Returns:
            File data as bytes or mmap object
        """
        file_path = Path(file_path)
        file_size = file_path.stat().st_size
        
        # For small files, just read the whole thing
        if file_size <= self.max_memory_bytes:
            logger.info(f"Reading small file {file_path} ({file_size} bytes) into memory")
            with open(file_path, 'rb') as f:
                return f.read()
        
        # For large files, use memory mapping or streaming
        if self.use_mmap:
            logger.info(f"Memory-mapping large file {file_path} ({file_size} bytes)")
            f = open(file_path, 'rb')
            # Keep track of file object to close it later
            self.temp_files.append(f.name)
            return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Create a temporary file for streaming
        temp_fd, temp_path = tempfile.mkstemp(dir=self.temp_dir)
        self.temp_files.append(temp_path)
        
        logger.info(f"Streaming large file {file_path} to temporary file {temp_path}")
        
        chunk_size = chunk_size or (1024 * 1024)  # Default to 1MB chunks
        
        with open(file_path, 'rb') as src, os.fdopen(temp_fd, 'wb') as dst:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                dst.write(chunk)
        
        # Return a file object for the temporary file
        return open(temp_path, 'rb')
    
    def create_temp_file(self, prefix: str = "zpdr_") -> Tuple[BinaryIO, str]:
        """
        Create a temporary file for storing intermediate results.
        
        Args:
            prefix: Prefix for the temporary file name
            
        Returns:
            Tuple of (file object, file path)
        """
        fd, path = tempfile.mkstemp(prefix=prefix, dir=self.temp_dir)
        self.temp_files.append(path)
        
        logger.info(f"Created temporary file {path}")
        
        return os.fdopen(fd, 'w+b'), path
    
    def create_memory_efficient_buffer(self, 
                                      initial_size_mb: int = 10) -> Union[io.BytesIO, BinaryIO]:
        """
        Create a memory-efficient buffer for storing data.
        
        For small data, uses BytesIO in memory.
        For potentially large data, uses a temporary file.
        
        Args:
            initial_size_mb: Initial size estimate in MB
            
        Returns:
            BytesIO or file object
        """
        initial_size_bytes = initial_size_mb * 1024 * 1024
        
        if initial_size_bytes <= self.max_memory_bytes:
            logger.info(f"Creating in-memory buffer of {initial_size_mb}MB")
            return io.BytesIO()
        
        logger.info(f"Creating file-backed buffer for {initial_size_mb}MB")
        buffer, _ = self.create_temp_file()
        return buffer
    
    def optimize_binary_data(self, data: bytes) -> Union[bytes, BinaryIO]:
        """
        Optimize storage of binary data.
        
        For small data, keeps it in memory.
        For large data, writes to a temporary file.
        
        Args:
            data: Binary data to optimize
            
        Returns:
            Original data or file object
        """
        if len(data) <= self.max_memory_bytes:
            return data
        
        buffer, _ = self.create_temp_file()
        buffer.write(data)
        buffer.flush()
        buffer.seek(0)
        
        logger.info(f"Moved {len(data)} bytes from memory to temporary file")
        
        return buffer