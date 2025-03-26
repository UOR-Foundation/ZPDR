#!/usr/bin/env python3
"""
Parallel processing utilities for ZPDR.

This module provides parallel processing capabilities for ZPDR operations,
improving performance for large files and computationally intensive tasks.
"""

import os
import time
import multiprocessing
from typing import List, Tuple, Dict, Any, Callable, Union, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """
    Result of a parallel processing operation.
    """
    success: bool
    chunk_id: int
    data: Any
    error: Optional[str] = None
    processing_time: float = 0.0

class ParallelProcessor:
    """
    Utility for parallel processing of data in ZPDR.
    """
    
    def __init__(self, 
                 max_workers: Optional[int] = None, 
                 use_processes: bool = True,
                 chunk_size: int = 1024 * 1024):
        """
        Initialize the parallel processor.
        
        Args:
            max_workers: Maximum number of worker processes/threads
            use_processes: Whether to use processes (True) or threads (False)
            chunk_size: Size of chunks for file processing
        """
        self.max_workers = max_workers or os.cpu_count() or 2
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        logger.info(f"Initialized ParallelProcessor with {self.max_workers} workers")
    
    def process_in_parallel(self, 
                           items: List[Any], 
                           process_func: Callable[[Any], Any]) -> List[Any]:
        """
        Process a list of items in parallel.
        
        Args:
            items: List of items to process
            process_func: Function to apply to each item
            
        Returns:
            List of processed items
        """
        start_time = time.time()
        
        logger.info(f"Processing {len(items)} items in parallel")
        
        # If using processes, need to check if we can pickle the function
        if self.use_processes:
            try:
                import pickle
                pickle.dumps(process_func)
            except:
                logger.warning("Function cannot be pickled, falling back to thread-based parallelism")
                self.use_processes = False
        
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        results = []
        
        # Process items in parallel
        with executor_class(max_workers=self.max_workers) as executor:
            if self.use_processes:
                # For processes, we need a different approach since we can't pass the process_func directly
                futures = []
                for i, item in enumerate(items):
                    # Apply the function here and submit only the processing of the result
                    try:
                        data = process_func(item)
                        futures.append(executor.submit(lambda x: ProcessingResult(
                            success=True, 
                            chunk_id=x[0], 
                            data=x[1], 
                            processing_time=0.0), 
                            (i, data)))
                    except Exception as e:
                        logger.error(f"Error processing item {i}: {e}")
                        results.append(ProcessingResult(
                            success=False,
                            chunk_id=i,
                            data=None,
                            error=str(e),
                            processing_time=0.0
                        ))
            else:
                # For threads, we can pass the function directly
                futures = [executor.submit(self._process_item, process_func, item, i) 
                          for i, item in enumerate(items)]
            
            # Collect results as they complete
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel processing: {e}")
        
        # Sort results by original index
        results.sort(key=lambda x: x.chunk_id)
        
        elapsed = time.time() - start_time
        logger.info(f"Parallel processing completed in {elapsed:.2f} seconds")
        
        # Extract data from results
        return [r.data for r in results if r.success]
    
    def _process_item(self, func: Callable, item: Any, item_id: int) -> ProcessingResult:
        """
        Process a single item and track performance.
        
        Args:
            func: Function to apply to the item
            item: Item to process
            item_id: Identifier for the item
            
        Returns:
            ProcessingResult containing the result or error
        """
        start_time = time.time()
        
        try:
            result = func(item)
            processing_time = time.time() - start_time
            return ProcessingResult(
                success=True,
                chunk_id=item_id,
                data=result,
                processing_time=processing_time
            )
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Error processing item {item_id}: {e}")
            return ProcessingResult(
                success=False,
                chunk_id=item_id,
                data=None,
                error=str(e),
                processing_time=processing_time
            )
    
    def process_file_in_chunks(self, 
                              file_path: Union[str, Path], 
                              process_chunk_func: Callable[[bytes, int], Any]) -> List[Any]:
        """
        Process a file in chunks in parallel.
        
        Args:
            file_path: Path to the file to process
            process_chunk_func: Function to process each chunk
            
        Returns:
            List of processing results for each chunk
        """
        file_path = Path(file_path)
        file_size = file_path.stat().st_size
        
        logger.info(f"Processing file {file_path} ({file_size} bytes) in chunks of {self.chunk_size} bytes")
        
        # Prepare chunks
        chunks = []
        with open(file_path, 'rb') as f:
            chunk_id = 0
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                chunks.append((chunk, chunk_id))
                chunk_id += 1
        
        logger.info(f"File divided into {len(chunks)} chunks")
        
        # For each chunk, apply the function directly
        results = []
        for chunk_data, chunk_id in chunks:
            try:
                result = process_chunk_func(chunk_data, chunk_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {e}")
        
        return results
    
    def merge_chunk_results(self, 
                           results: List[Any],
                           merge_func: Callable[[List[Any]], Any]) -> Any:
        """
        Merge results from chunk processing.
        
        Args:
            results: List of chunk processing results
            merge_func: Function to merge the results
            
        Returns:
            Merged result
        """
        logger.info(f"Merging {len(results)} chunk results")
        return merge_func(results)