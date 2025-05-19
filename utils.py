import json
import os
import logging

def save_json(data, output_path):
    logger = logging.getLogger(__name__)
    
    try:
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Data saved successfully to {output_path}")
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {str(e)}")
        raise

def save_csv(data, output_path):
    import csv
    logger = logging.getLogger(__name__)
    
    try:
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        
        flattened_data = []
        for item in data:
            flattened_item = item.copy()
            
            
            entities = flattened_item.pop("named_entities", {})
            for entity_type, entity_list in entities.items():
                flattened_item[f"{entity_type}"] = "|".join(entity_list)
            
            flattened_data.append(flattened_item)
        
        
        if flattened_data:
            fieldnames = flattened_data[0].keys()
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_data)
            
            logger.info(f"Data saved successfully to {output_path}")
        else:
            logger.warning("No data to save to CSV")
    except Exception as e:
        logger.error(f"Error saving data to {output_path}: {str(e)}")
        raise
