import pdfplumber  # PDF parsing and text extraction library
import os          # File system operations
import logging     # Logging framework for diagnostics
import re          # Regular expressions for pattern matching
from collections import defaultdict  # Used for tracking frequency distributions

#===============================================================
# Document Reader - Handles PDF extraction and pre-processing
#===============================================================

class DocumentReader:
    """Handles PDF document reading, text extraction and layout analysis.
    
    This class is responsible for extracting text from PDF documents while
    preserving structure information. It handles multi-column layouts,
    detects and removes headers/footers, and provides positional information
    for mapping text back to original page locations.
    
    Key capabilities:
    - Extract text with layout awareness
    - Detect and handle multi-column layouts
    - Identify and remove headers and footers
    - Provide page mapping for extracted text
    - Handle various PDF formats and encodings
    """
    
    def __init__(self):
        """Initialize the document reader with default settings.
        
        Sets up logging and initializes tracking data structures for
        page metadata, layout information, and header/footer detection.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.page_numbers = {}  # Maps text positions to original page numbers
        self.header_footer_candidates = []  # Tracks potential headers/footers
        
        # Document layout information, populated during extraction
        self.layout_info = {
            'multi_column': False,  # Whether document has multiple columns
            'columns_count': 1,     # Number of columns detected (1 = single column)
            'has_header': False,    # Whether document has repeating headers
            'has_footer': False,    # Whether document has repeating footers
            'toc_pages': []         # Pages that appear to be table of contents
        }
    
    def extract_text_from_pdf(self, pdf_path, remove_headers_footers=True, detect_columns=True):
        """Extract text from PDF document with layout awareness.
        
        This is the primary method for extracting text from PDF files.
        It performs multiple processing steps including:
        1. Loading and parsing the PDF using pdfplumber
        2. Detecting document layout (columns, headers/footers)
        3. Extracting text with positional information
        4. Cleaning and normalizing the extracted content
        
        Args:
            pdf_path: Path to the PDF file to process
            remove_headers_footers: Whether to detect and remove repeating headers/footers
            detect_columns: Whether to analyze for multi-column layouts
            
        Returns:
            tuple: (full_text, page_metadata, layout_info)
                - full_text: Extracted text with preserved structure
                - page_metadata: Mapping between text positions and source pages
                - layout_info: Dictionary of detected layout properties
                
        Raises:
            FileNotFoundError: If pdf_path doesn't exist
            ValueError: If PDF is corrupted or cannot be processed
        """
        # Validate input file exists
        if not os.path.exists(pdf_path):
            self.logger.error(f"File not found: {pdf_path}")
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")
        
        try:
            # Initialize output structures
            full_text = ""
            page_metadata = []  # Will store position-to-page mappings
            
            self.logger.info("Optimized single-pass extraction")
            
            with pdfplumber.open(pdf_path) as pdf:
                self.logger.info(f"Processing PDF with {len(pdf.pages)} pages")
                
                if remove_headers_footers and len(pdf.pages) > 2:
                    sample_indices = [0, min(1, len(pdf.pages)-1), min(len(pdf.pages)-1, 2)]
                    
                    for i in sample_indices:
                        page = pdf.pages[i]
                        text_blocks = self._extract_text_blocks(page, simplified=True)
                        self._identify_header_footer_candidates(page, text_blocks, i)
                    
                    if remove_headers_footers:
                        self._finalize_header_footer_identification(len(pdf.pages))
                
                
                # Column detection phase - only run if requested
                if detect_columns:
                    # Performance optimization: Only check first few pages
                    # Most documents maintain same column layout throughout
                    # Checking all pages would be wasteful for large documents
                    max_check = min(3, len(pdf.pages))
                    
                    # Sample early pages for column structure detection
                    for i in range(max_check):
                        page = pdf.pages[i]
                        # Extract text blocks using simplified mode for speed
                        text_blocks = self._extract_text_blocks(page, simplified=True)
                        # Analyze spatial distribution to detect columns
                        columns = self._detect_columns(page, text_blocks, simplified=True)
                        
                        # If multiple columns are detected, update the layout info
                        if len(columns) > 1:
                            self.layout_info['multi_column'] = True
                            # Update to highest column count found (some pages may vary)
                            self.layout_info['columns_count'] = max(self.layout_info['columns_count'], len(columns))
                            break  # Early exit once we've confirmed multiple columns
                
                
                #--------------------------------------------------------------------
                # Main extraction loop - Process each page individually
                #--------------------------------------------------------------------
                for i, page in enumerate(pdf.pages):
                    # Extract page text using the appropriate method based on layout detection
                    if self.layout_info['multi_column'] and detect_columns:
                        # For multi-column pages, use specialized column-aware extraction
                        text_blocks = self._extract_text_blocks(page, simplified=True)
                        page_text = self._extract_text_from_columns(page, text_blocks)
                    else:
                        # For simple layouts, use pdfplumber's built-in extraction
                        page_text = page.extract_text()
                    
                    # Only process pages that contain text
                    if page_text:
                        # Remove headers/footers if requested and detected
                        if remove_headers_footers:
                            page_text = self._remove_headers_footers(page_text, i)
                        
                        # Track character positions to map back to original pages later
                        # This is crucial for maintaining document context during segmentation
                        start_index = len(full_text)
                        full_text += page_text + "\n\n"  # Add double newline as page separator
                        end_index = len(full_text)
                        
                        # Store page metadata for later reference
                        # This enables mappings from text positions back to source pages
                        page_metadata.append({
                            'page_number': i + 1,  # 1-indexed page numbers for output
                            'start_index': start_index,  # Character position where page starts
                            'end_index': end_index,      # Character position where page ends
                            'text_length': len(page_text) # Length of text content in this page
                        })
            
            #--------------------------------------------------------------------
            # Finalize layout detection and metadata
            #--------------------------------------------------------------------
            # Update layout info with header/footer detection results
            self.layout_info['has_header'] = any(h['is_header'] for h in self.header_footer_candidates)
            self.layout_info['has_footer'] = any(h['is_footer'] for h in self.header_footer_candidates)
            
            # Log extraction statistics
            self.logger.info(f"Extraction complete: {len(full_text)} characters from {len(page_metadata)} pages")
            
            # Return the extracted text, layout information, and page mapping metadata
            # This provides a complete picture of the document for downstream processing
            return full_text.strip(), self.layout_info, page_metadata
            
        #--------------------------------------------------------------------
        # Error handling
        #--------------------------------------------------------------------
        except Exception as e:
            # Log the error with details
            self.logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
            # Re-raise to allow calling code to handle the failure
            raise  # Preserves original stack trace
    
    def _extract_text_blocks(self, page, simplified=False):
        """Extract text blocks from a PDF page with layout information.
        
        This method extracts text from a PDF page while preserving spatial information.
        It can operate in two modes:
        1. Simplified: Fast extraction treating the entire page as a single block
        2. Detailed: Extracts individual words and groups them into lines/blocks
           based on their spatial proximity
        
        Args:
            page: A pdfplumber page object to extract text from
            simplified: Whether to use simplified extraction (faster but less precise)
            
        Returns:
            List of text block dictionaries with text content and bounding box info
        """
        try:
            # Simplified mode - fast extraction for initial analysis
            if simplified:
                text = page.extract_text()
                if not text:
                    return []  # Return empty list for pages without text
                # Return the entire page as a single text block
                return [{
                    "text": text,  # All text content
                    "bbox": (0, 0, page.width, page.height)  # Page dimensions as bbox
                }]
            
            # Detailed mode - extract individual words with position information
            # Using tolerances to handle slight alignment variations
            page_words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
            
            # Handle empty pages
            if not page_words:
                return []
            
            # Group words into lines based on vertical position
            lines = {}  # Dictionary to group words by vertical position
            line_tolerance = 5  # Pixels tolerance for line grouping
            
            for word in page_words:
                y_key = round(word["top"] / line_tolerance) * line_tolerance
                
                if y_key not in lines:
                    lines[y_key] = []
                lines[y_key].append(word)
            
            text_blocks = []
            
            for y_key in sorted(lines.keys()):
                line_words = sorted(lines[y_key], key=lambda w: w["x0"])
                
                if not line_words:
                    continue
                    
                x0 = min(w["x0"] for w in line_words)
                x1 = max(w["x1"] for w in line_words)
                y0 = min(w["top"] for w in line_words)
                y1 = max(w["bottom"] for w in line_words)
                
                text = " ".join(w["text"] for w in line_words)
                
                if text.strip():
                    text_blocks.append({
                        "text": text,
                        "bbox": (x0, y0, x1, y1)
                    })
            
            if len(text_blocks) > 100:
                text_blocks.sort(key=lambda b: (b["bbox"][2]-b["bbox"][0])*(b["bbox"][3]-b["bbox"][1]), reverse=True)
                text_blocks = text_blocks[:100]
                
            return text_blocks
            
        except Exception as e:
            self.logger.warning(f"Error extracting text blocks: {str(e)}")
            try:
                text = page.extract_text()
                if text:
                    return [{
                        "text": text,
                        "bbox": (0, 0, page.width, page.height)
                    }]
            except:
                pass
            return []
        
        return blocks
    
    def _detect_page_number(self, page, page_idx, text_blocks):
        page_height = page.height
        page_width = page.width
        
        regions = [
            (page_width * 0.4, page_height * 0.9, page_width * 0.6, page_height),
            (page_width * 0.4, 0, page_width * 0.6, page_height * 0.1),
            (page_width * 0.8, page_height * 0.9, page_width, page_height),
            (0, page_height * 0.9, page_width * 0.2, page_height)
        ]
        
        for block in text_blocks:
            x0, top, x1, bottom = block['bbox']
            
            for region in regions:
                r_x0, r_top, r_x1, r_bottom = region
                if (x0 >= r_x0 and x1 <= r_x1 and top >= r_top and bottom <= r_bottom):
                    text = block['text'].strip()
                    if len(text) < 10 and any(c.isdigit() for c in text):
                        numbers = re.findall(r'\d+', text)
                        if numbers:
                            return int(numbers[0])
        
        return None
    
    def _identify_header_footer_candidates(self, page, text_blocks, page_idx):
        page_height = page.height
        
        for block in text_blocks:
            x0, top, x1, bottom = block['bbox']
            text = block['text'].strip()
            
            if len(text) < 3:
                continue
            
            if top < page_height * 0.1:
                self.header_footer_candidates.append({
                    'text': text,
                    'page': page_idx,
                    'position': 'header',
                    'bbox': (x0, top, x1, bottom)
                })
            
            if bottom > page_height * 0.9:
                self.header_footer_candidates.append({
                    'text': text,
                    'page': page_idx,
                    'position': 'footer',
                    'bbox': (x0, top, x1, bottom)
                })

    def _finalize_header_footer_identification(self, total_pages):
        header_text_counts = defaultdict(list)
        footer_text_counts = defaultdict(list)
        
        for candidate in self.header_footer_candidates:
            if candidate['position'] == 'header':
                header_text_counts[candidate['text']].append(candidate['page'])
            else:  
                footer_text_counts[candidate['text']].append(candidate['page'])
        
        min_occurrences = max(2, total_pages // 5)  
        
        self.headers = []
        self.footers = []
        
        for text, pages in header_text_counts.items():
            if len(pages) >= min_occurrences:
                self.headers.append(text)
                self.layout_info['has_header'] = True
        
        for text, pages in footer_text_counts.items():
            if len(pages) >= min_occurrences:
                self.footers.append(text)
                self.layout_info['has_footer'] = True
        
        for candidate in self.header_footer_candidates:
            candidate['is_header'] = (candidate['position'] == 'header' and candidate['text'] in self.headers)
            candidate['is_footer'] = (candidate['position'] == 'footer' and candidate['text'] in self.footers)
        
        self.logger.info(f"Identified {len(self.headers)} potential headers and {len(self.footers)} potential footers")
    
    def _remove_headers_footers(self, text, page_idx):
        if not text:
            return text
        
        headers = [candidate['text'] for candidate in self.header_footer_candidates if candidate.get('is_header', False)]
        footers = [candidate['text'] for candidate in self.header_footer_candidates if candidate.get('is_footer', False)]
        
        if not headers and not footers:
            return text
        
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            keep_line = True
            
            for header in headers:
                if header in line or self._similarity_check(header, line):
                    keep_line = False
                    break
            
            if keep_line:
                for footer in footers:
                    if footer in line or self._similarity_check(footer, line):
                        keep_line = False
                        break
            
            if keep_line:
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def _similarity_check(self, text1, text2):
        t1 = re.sub(r'\s+', '', text1.lower())
        t2 = re.sub(r'\s+', '', text2.lower())
        
        if t1 in t2 or t2 in t1:
            return True
        
        if len(t1) > 10 and len(t2) > 10:  
            shorter = t1 if len(t1) <= len(t2) else t2
            longer = t2 if len(t1) <= len(t2) else t1
            
            matches = sum(1 for c in shorter if c in longer)
            similarity = matches / len(shorter)
            
            return similarity > 0.8  
    
        return False
    
    def _detect_columns(self, page, text_blocks, simplified=False):
        if not text_blocks:
            return []
        
        page_width = float(page.width)
        
        if simplified:
            if len(text_blocks) == 1 and text_blocks[0].get('width', 0) > page_width * 0.8:
                return [text_blocks]
            
            left_blocks = []
            right_blocks = []
            middle_blocks = []
            
            for block in text_blocks:
                center_x = block.get('x_center', block['bbox'][0] + (block['bbox'][2] - block['bbox'][0])/2)
                if center_x < page_width * 0.33:
                    left_blocks.append(block)
                elif center_x > page_width * 0.67:
                    right_blocks.append(block)
                else:
                    middle_blocks.append(block)
            
            if left_blocks and right_blocks and not middle_blocks:
                return [left_blocks, right_blocks]
            
            if len(left_blocks) > 1 and len(middle_blocks) > 1 and len(right_blocks) > 1:
                return [left_blocks, middle_blocks, right_blocks]
            
            return [text_blocks]
        
        columns = []
        x_positions = [(block['bbox'][0], block['bbox'][2]) for block in text_blocks]
        x_starts = [pos[0] for pos in x_positions]
        
        sorted_starts = sorted(x_starts)
        gaps = [sorted_starts[i+1] - sorted_starts[i] for i in range(len(sorted_starts)-1)]
        
        if not gaps:
            return [text_blocks]  
        
        avg_gap = sum(gaps) / len(gaps)
        std_dev = (sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5
        significant_gap = avg_gap + std_dev * 1.5
        
        column_boundaries = [0]  
        
        for i, gap in enumerate(gaps):
            if gap > significant_gap and sorted_starts[i+1] - sorted_starts[0] > page_width * 0.15:
                column_boundaries.append((sorted_starts[i] + sorted_starts[i+1]) / 2)  
        
        column_boundaries.append(page_width)  
        
        if len(column_boundaries) > 2:
            columns = [[] for _ in range(len(column_boundaries)-1)]
            
            for block in text_blocks:
                x0 = block['bbox'][0]
                
                for i in range(len(column_boundaries)-1):
                    if x0 >= column_boundaries[i] and x0 < column_boundaries[i+1]:
                        columns[i].append(block)
                        break
            
            columns = [col for col in columns if col]
            
            if len(columns) > 1:
                return columns
        
        return [text_blocks]  
    
    def _extract_text_from_columns(self, page, text_blocks):
        columns = self._detect_columns(page, text_blocks)
        
        if len(columns) <= 1:
            return page.extract_text()
        
        column_texts = []
        for column in columns:
            sorted_blocks = sorted(column, key=lambda b: b['bbox'][1])
            column_text = '\n'.join(block['text'] for block in sorted_blocks)
            column_texts.append(column_text)
        
        return '\n\n'.join(column_texts)
    
    def _detect_toc_pages(self, pdf):
        toc_pages = []
        
        for i, page in enumerate(pdf.pages[:min(10, len(pdf.pages))]):
            text = page.extract_text()
            if not text:
                continue
            
            toc_indicators = [
                r'\bcontents\b',
                r'\btable of contents\b',
                r'\bindex\b',
                r'\btoc\b'
            ]
            
            page_number_pattern = r'\b\d+[\s\.]+\d+\b'
            
            lines = text.lower().split('\n')
            toc_like_lines = 0
            
            for line in lines:
                if re.search(page_number_pattern, line):
                    toc_like_lines += 1
            
            if any(re.search(pattern, text.lower()) for pattern in toc_indicators) or toc_like_lines > 5:
                toc_pages.append(i)
        
        return toc_pages
