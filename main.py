#========================================================================
# Document Processor - Command-line interface for PDF processing pipeline
#========================================================================

# Standard library imports
import argparse  # Command-line argument parsing
import logging   # Logging for diagnostics and tracking
import os        # File system operations
import sys       # System-specific parameters and functions

# Local application imports
from document_reader import DocumentReader    # PDF extraction module
from segmentation import DocumentSegmenter    # Document structure analysis
from ner_processor import NERProcessor        # Named entity recognition
from utils import save_json, save_csv         # Output formatting utilities


def setup_logging():
    """Configure application logging for both console and file output.
    
    Sets up a dual logging configuration that:  
    1. Outputs all logs to the console (stdout) for immediate feedback
    2. Writes all logs to a file for later inspection and debugging
    
    The log file is overwritten on each run to prevent excessive log growth.
    """
    logging.basicConfig(
        level=logging.INFO,  # Set minimum log level to INFO
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Detailed log format
        handlers=[
            logging.StreamHandler(stream=sys.stdout),  # Console output
            logging.FileHandler('document_processor.log', mode='w')  # File output (overwrite existing)
        ]
    )

def main():
    """Command-line entry point for the document processing pipeline.
    
    This is the main entry point for the PDF document processing application when run
    from the command line. It handles argument parsing, executes the processing pipeline,
    and outputs results. The function can be configured via command-line arguments.
    
    Pipeline steps executed:
    1. PDF text extraction with layout analysis
    2. Document segmentation and structure analysis 
    3. Named entity recognition and classification
    4. Results output in specified format (JSON or CSV)
    """
    # Initialize logging system
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description='Process PDF documents for segmentation and NER.')
    parser.add_argument('--input', '-i', required=True, help='Path to the input PDF file')
    parser.add_argument('--output', '-o', required=True, help='Path to save the output file')
    parser.add_argument('--format', '-f', choices=['json', 'csv'], default='json', 
                        help='Output format (default: json)')
    parser.add_argument('--spacy-model', default='en_core_web_lg', 
                       help='SpaCy model to use (default: en_core_web_lg)')
    args = parser.parse_args()
    
    try:
        # Log the start of the pipeline with the input file path
        logger.info(f"Starting document processing pipeline for: {args.input}")
        
        #--------------------------------------------------------------------
        # STEP 1: PDF TEXT EXTRACTION
        #--------------------------------------------------------------------
        logger.info("Step 1: Extracting text from PDF")
        reader = DocumentReader()
        # Extract text while analyzing layout for headers, footers, columns, etc.
        text = reader.extract_text_from_pdf(args.input)
        
        #--------------------------------------------------------------------
        # STEP 2: DOCUMENT SEGMENTATION
        #--------------------------------------------------------------------
        logger.info("Step 2: Segmenting document")
        segmenter = DocumentSegmenter()
        # Build hierarchical segments from document structure
        segments = segmenter.segment_document(text)
        
        #--------------------------------------------------------------------
        # STEP 3: NAMED ENTITY RECOGNITION
        #--------------------------------------------------------------------
        logger.info("Step 3: Processing named entities")
        # Initialize NER processor with specified SpaCy model
        ner = NERProcessor(model=args.spacy_model)
        # Extract entities from each segment with context awareness
        processed_segments = ner.process_segments(segments)
        
        #--------------------------------------------------------------------
        # STEP 4: OUTPUT GENERATION
        #--------------------------------------------------------------------
        logger.info(f"Step 4: Saving results as {args.format.upper()}")
        # Save processed data in requested format (JSON or CSV)
        if args.format.lower() == 'json':
            save_json(processed_segments, args.output)  # Hierarchical JSON
        else:
            save_csv(processed_segments, args.output)   # Flattened CSV
        
        # Log and print success message
        logger.info("Document processing complete")
        print(f"Document processing complete. Results saved to: {args.output}")
        
        #--------------------------------------------------------------------
        # Generate sample output for user feedback
        #--------------------------------------------------------------------
        # Only show a sample if we have segments to display
        if processed_segments:
            print("\nSample output (first segment):")
            
            # Create a truncated copy of the first segment for display
            sample = processed_segments[0].copy()
            
            # Truncate segment text to prevent excessive output
            if len(sample["segment_text"]) > 200:
                sample["segment_text"] = sample["segment_text"][:200] + "..."
            
            # Truncate entity lists to show only first 5 of each type
            for entity_type, entities in sample["named_entities"].items():
                if len(entities) > 5:
                    sample["named_entities"][entity_type] = entities[:5] + ["..."]
            
            # Pretty-print the sample as formatted JSON
            import json
            print(json.dumps(sample, indent=2))
            
    #--------------------------------------------------------------------
    # ERROR HANDLING
    #--------------------------------------------------------------------
    except Exception as e:
        # Log full exception details including stack trace
        logger.error(f"Error in document processing pipeline: {str(e)}", exc_info=True)
        # Print user-friendly error message to console
        print(f"Error: {str(e)}")
        # Exit with non-zero status to indicate failure
        sys.exit(1)

if __name__ == "__main__":
    main()
