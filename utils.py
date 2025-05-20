#================================================================
# Utility Functions - Data output and format conversion utilities
#================================================================

import json       # For JSON serialization/deserialization
import os         # For file system operations 
import logging    # For diagnostic logging


def save_json(data, output_path):
    """Save structured data to a JSON file with proper formatting.
    
    Creates necessary directories if they don't exist and handles Unicode
    characters properly in the output file. The JSON is pretty-printed
    with indentation for better readability.
    
    Args:
        data: Dictionary or list to serialize to JSON
        output_path: Path where the JSON file should be saved
        
    Raises:
        IOError: If the file cannot be written
        TypeError: If the data cannot be serialized to JSON
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write JSON with proper formatting and encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)  # Pretty-print with Unicode support
        
        logger.info(f"Data saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {str(e)}")
        raise  # Re-raise for proper error handling

def save_csv(data, output_path):
    """Save processed document data to CSV format.
    
    Converts the hierarchical document structure (with nested entities) into
    a flattened CSV format. This involves:
    1. Flattening the nested entity structure
    2. Concatenating entity lists with pipe (|) separators
    3. Writing headers and rows to CSV with proper encoding
    
    Args:
        data: List of segment dictionaries from document processing
        output_path: Path where the CSV file should be saved
        
    Raises:
        IOError: If the file cannot be written
        KeyError: If the data structure is not as expected
    """
    import csv
    logger = logging.getLogger(__name__)
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Transform hierarchical structure to flat rows for CSV format
        # Each segment becomes a row with entities flattened to columns
        flattened_data = []
        for item in data:
            flattened_item = item.copy()  # Copy to avoid modifying original
            
            # Extract and flatten named entities (which are nested dictionaries)
            # Convert each entity list to a pipe-separated string for CSV compatibility
            entities = flattened_item.pop("named_entities", {})
            for entity_type, entity_list in entities.items():
                flattened_item[f"{entity_type}"] = "|".join(entity_list)
            
            flattened_data.append(flattened_item)
        
        # Write CSV file if we have data to write
        if flattened_data:
            # Extract field names from first item's keys
            fieldnames = flattened_data[0].keys()
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()  # Write CSV header row
                writer.writerows(flattened_data)  # Write all data rows
            
            logger.info(f"Data saved successfully to {output_path}")
        else:
            logger.warning("No data to save to CSV")
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {str(e)}")
        raise  # Re-raise for proper error handling
