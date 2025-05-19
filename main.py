import argparse
import logging
import os
import sys
from document_reader import DocumentReader
from segmentation import DocumentSegmenter
from ner_processor import NERProcessor
from utils import save_json, save_csv

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler('document_processor.log', mode='w')
        ]
    )

def main():
    """Main function to run the document processing pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process PDF documents for segmentation and NER.')
    parser.add_argument('--input', '-i', required=True, help='Path to the input PDF file')
    parser.add_argument('--output', '-o', required=True, help='Path to save the output file')
    parser.add_argument('--format', '-f', choices=['json', 'csv'], default='json', help='Output format (default: json)')
    parser.add_argument('--spacy-model', default='en_core_web_lg', help='SpaCy model to use (default: en_core_web_lg)')
    args = parser.parse_args()
    
    try:
        logger.info(f"Starting document processing pipeline for: {args.input}")
        
        # Step 1: Extract text from PDF
        logger.info("Step 1: Extracting text from PDF")
        reader = DocumentReader()
        text = reader.extract_text_from_pdf(args.input)
        
        # Step 2: Segment the document
        logger.info("Step 2: Segmenting document")
        segmenter = DocumentSegmenter()
        segments = segmenter.segment_document(text)
        
        # Step 3: Process named entities
        logger.info("Step 3: Processing named entities")
        ner = NERProcessor(model=args.spacy_model)
        processed_segments = ner.process_segments(segments)
        
        # Step 4: Save the results
        logger.info(f"Step 4: Saving results as {args.format.upper()}")
        if args.format.lower() == 'json':
            save_json(processed_segments, args.output)
        else:
            save_csv(processed_segments, args.output)
        
        logger.info("Document processing complete")
        print(f"Document processing complete. Results saved to: {args.output}")
        
        # Print a sample of the first segment for verification
        if processed_segments:
            print("\nSample output (first segment):")
            
            # Pretty print the first segment (limited output for console)
            sample = processed_segments[0].copy()
            if len(sample["segment_text"]) > 200:
                sample["segment_text"] = sample["segment_text"][:200] + "..."
            
            for entity_type, entities in sample["named_entities"].items():
                if len(entities) > 5:
                    sample["named_entities"][entity_type] = entities[:5] + ["..."]
            
            import json
            print(json.dumps(sample, indent=2))
            
    except Exception as e:
        logger.error(f"Error in document processing pipeline: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
