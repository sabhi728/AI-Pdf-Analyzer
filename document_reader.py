import pdfplumber
import os
import logging
import re
from collections import defaultdict

class DocumentReader:
    """Enhanced class for reading and extracting text from PDF documents with advanced layout analysis."""
    
    def __init__(self):
        """Initialize the document reader."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.page_numbers = {}
        self.header_footer_candidates = []
        self.layout_info = {
            'multi_column': False,
            'columns_count': 1,
            'has_header': False,
            'has_footer': False,
            'toc_pages': []
        }
    
    def extract_text_from_pdf(self, pdf_path, remove_headers_footers=True, detect_columns=True):
        """
        Extract text from a PDF document with optimized layout analysis (single-pass approach).
        
        Args:
            pdf_path (str): Path to the PDF file.
            remove_headers_footers (bool): Whether to attempt to remove headers and footers.
            detect_columns (bool): Whether to attempt to detect and process columns.
            
        Returns:
            tuple: (extracted_text, layout_info, page_metadata)
                - extracted_text (str): The full extracted text.
                - layout_info (dict): Information about the document layout.
                - page_metadata (list): Metadata for each page including page numbers and positions.
        """
        if not os.path.exists(pdf_path):
            self.logger.error(f"File not found: {pdf_path}")
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")
        
        try:
            # Initialize storage for extracted text and metadata
            full_text = ""
            page_metadata = []
            
            # Single-pass optimization: analyze and extract in one go
            self.logger.info("Optimized single-pass extraction")
            
            with pdfplumber.open(pdf_path) as pdf:
                self.logger.info(f"Processing PDF with {len(pdf.pages)} pages")
                
                # Process a sample of pages for header/footer detection if needed
                if remove_headers_footers and len(pdf.pages) > 2:
                    # Sample pages for faster header/footer detection
                    sample_indices = [0, min(1, len(pdf.pages)-1), min(len(pdf.pages)-1, 2)]
                    
                    for i in sample_indices:
                        page = pdf.pages[i]
                        text_blocks = self._extract_text_blocks(page, simplified=True)
                        self._identify_header_footer_candidates(page, text_blocks, i)
                    
                    if remove_headers_footers:
                        self._finalize_header_footer_identification(len(pdf.pages))
                
                # Fast check for multi-column layout on first few pages
                if detect_columns:
                    max_check = min(3, len(pdf.pages))
                    for i in range(max_check):
                        page = pdf.pages[i]
                        text_blocks = self._extract_text_blocks(page, simplified=True)
                        columns = self._detect_columns(page, text_blocks, simplified=True)
                        if len(columns) > 1:
                            self.layout_info['multi_column'] = True
                            self.layout_info['columns_count'] = max(self.layout_info['columns_count'], len(columns))
                            break
                
                # Fast extract text from all pages
                for i, page in enumerate(pdf.pages):
                    # Extract text based on layout analysis
                    if self.layout_info['multi_column'] and detect_columns:
                        text_blocks = self._extract_text_blocks(page, simplified=True)
                        page_text = self._extract_text_from_columns(page, text_blocks)
                    else:
                        # Faster direct extraction
                        page_text = page.extract_text()
                    
                    if page_text:
                        # Remove headers and footers if detected
                        if remove_headers_footers:
                            page_text = self._remove_headers_footers(page_text, i)
                        
                        # Store metadata about page text position
                        start_index = len(full_text)
                        full_text += page_text + "\n\n"
                        end_index = len(full_text)
                        
                        page_metadata.append({
                            'page_number': i + 1,  # Use sequential number for speed
                            'start_index': start_index,
                            'end_index': end_index,
                            'text_length': len(page_text)
                        })
            
            # Update layout info based on analysis
            self.layout_info['has_header'] = any(h['is_header'] for h in self.header_footer_candidates)
            self.layout_info['has_footer'] = any(h['is_footer'] for h in self.header_footer_candidates)
            
            self.logger.info(f"Extraction complete: {len(full_text)} characters from {len(page_metadata)} pages")
            return full_text.strip(), self.layout_info, page_metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
            raise
    
    def _extract_text_blocks(self, page, simplified=False):
        """Extract text blocks with their bounding boxes.
        
        Args:
            page: The PDF page object
            simplified: If True, use a faster algorithm with fewer features
        """
        try:
            # Extremely fast mode for basic extraction
            if simplified:
                # Direct extraction instead of word-by-word processing
                text = page.extract_text()
                if not text:
                    return []
                    
                # Just return the page as a single text block for header/footer detection
                return [{
                    "text": text,
                    "bbox": (0, 0, page.width, page.height)
                }]
            
            # Get words from the page with their bounding boxes using optimized settings
            page_words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
            
            if not page_words:
                return []
                
            # Fast line grouping using pre-allocated dictionary size
            lines = {}
            line_tolerance = 5  # Increased tolerance for fewer blocks
            
            # Group words by their y-position with rounding for faster line detection
            for word in page_words:
                # Round y position to nearest multiple of tolerance for better grouping
                y_key = round(word["top"] / line_tolerance) * line_tolerance
                
                if y_key not in lines:
                    lines[y_key] = []
                lines[y_key].append(word)
            
            # Create blocks from lines - much faster than paragraph processing
            text_blocks = []
            
            for y_key in sorted(lines.keys()):
                line_words = sorted(lines[y_key], key=lambda w: w["x0"])
                
                if not line_words:
                    continue
                    
                # Calculate bounding box
                x0 = min(w["x0"] for w in line_words)
                x1 = max(w["x1"] for w in line_words)
                y0 = min(w["top"] for w in line_words)
                y1 = max(w["bottom"] for w in line_words)
                
                # Join words with spaces
                text = " ".join(w["text"] for w in line_words)
                
                # Only add non-empty blocks
                if text.strip():
                    text_blocks.append({
                        "text": text,
                        "bbox": (x0, y0, x1, y1)
                    })
            
            # Limit number of blocks for performance
            if len(text_blocks) > 100:
                # Sort by area and keep largest blocks
                text_blocks.sort(key=lambda b: (b["bbox"][2]-b["bbox"][0])*(b["bbox"][3]-b["bbox"][1]), reverse=True)
                text_blocks = text_blocks[:100]
                
            return text_blocks
            
        except Exception as e:
            self.logger.warning(f"Error extracting text blocks: {str(e)}")
            # Fast fallback with minimal processing
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
        """Detect the page number within the page."""
        # Look for page numbers in expected positions (corners, center bottom/top)
        page_height = page.height
        page_width = page.width
        
        # Define regions where page numbers typically appear
        regions = [
            # Bottom center
            (page_width * 0.4, page_height * 0.9, page_width * 0.6, page_height),
            # Top center
            (page_width * 0.4, 0, page_width * 0.6, page_height * 0.1),
            # Bottom right
            (page_width * 0.8, page_height * 0.9, page_width, page_height),
            # Bottom left
            (0, page_height * 0.9, page_width * 0.2, page_height)
        ]
        
        for block in text_blocks:
            x0, top, x1, bottom = block['bbox']
            
            # Check if the block is in a typical page number region
            for region in regions:
                r_x0, r_top, r_x1, r_bottom = region
                if (x0 >= r_x0 and x1 <= r_x1 and top >= r_top and bottom <= r_bottom):
                    # Check if the text looks like a page number
                    text = block['text'].strip()
                    # Simple check: short text containing digits
                    if len(text) < 10 and any(c.isdigit() for c in text):
                        # Extract the number from text (might contain "Page X" or just "X")
                        numbers = re.findall(r'\d+', text)
                        if numbers:
                            return int(numbers[0])
        
        return None
    
    def _identify_header_footer_candidates(self, page, text_blocks, page_idx):
        """Identify potential headers and footers based on position."""
        page_height = page.height
        
        for block in text_blocks:
            x0, top, x1, bottom = block['bbox']
            text = block['text'].strip()
            
            # Skip very short or empty blocks
            if len(text) < 3:
                continue
            
            # Potential header (top 10% of page)
            if top < page_height * 0.1:
                self.header_footer_candidates.append({
                    'text': text,
                    'page': page_idx,
                    'position': 'header',
                    'bbox': (x0, top, x1, bottom)
                })
            
            # Potential footer (bottom 10% of page)
            if bottom > page_height * 0.9:
                self.header_footer_candidates.append({
                    'text': text,
                    'page': page_idx,
                    'position': 'footer',
                    'bbox': (x0, top, x1, bottom)
                })

    def _finalize_header_footer_identification(self, total_pages):
        """Analyze collected candidates to identify consistent headers and footers."""
        # Group candidates by their text content
        header_text_counts = defaultdict(list)
        footer_text_counts = defaultdict(list)
        
        for candidate in self.header_footer_candidates:
            if candidate['position'] == 'header':
                header_text_counts[candidate['text']].append(candidate['page'])
            else:  # footer
                footer_text_counts[candidate['text']].append(candidate['page'])
        
        # Headers and footers should appear on multiple pages
        min_occurrences = max(2, total_pages // 5)  # At least 2 or 20% of pages
        
        # Mark each candidate with is_header/is_footer flags
        self.headers = []
        self.footers = []
        
        # First loop to identify common headers and footers
        for text, pages in header_text_counts.items():
            if len(pages) >= min_occurrences:
                self.headers.append(text)
                self.layout_info['has_header'] = True
        
        for text, pages in footer_text_counts.items():
            if len(pages) >= min_occurrences:
                self.footers.append(text)
                self.layout_info['has_footer'] = True
        
        # Second loop to mark candidates with is_header/is_footer flags
        for candidate in self.header_footer_candidates:
            candidate['is_header'] = (candidate['position'] == 'header' and candidate['text'] in self.headers)
            candidate['is_footer'] = (candidate['position'] == 'footer' and candidate['text'] in self.footers)
        
        self.logger.info(f"Identified {len(self.headers)} potential headers and {len(self.footers)} potential footers")
    
    def _remove_headers_footers(self, text, page_idx):
        """Remove identified headers and footers from the page text."""
        if not text:
            return text
        
        # Extract headers and footers from the candidates list
        headers = [candidate['text'] for candidate in self.header_footer_candidates if candidate.get('is_header', False)]
        footers = [candidate['text'] for candidate in self.header_footer_candidates if candidate.get('is_footer', False)]
        
        # If no headers or footers found, return the original text
        if not headers and not footers:
            return text
        
        lines = text.split('\n')
        clean_lines = []
        
        # Process each line
        for line in lines:
            keep_line = True
            
            # Check if this line matches any header
            for header in headers:
                if header in line or self._similarity_check(header, line):
                    keep_line = False
                    break
            
            # Check if this line matches any footer
            if keep_line:
                for footer in footers:
                    if footer in line or self._similarity_check(footer, line):
                        keep_line = False
                        break
            
            if keep_line:
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def _similarity_check(self, text1, text2):
        """Check if two texts are similar (for fuzzy header/footer matching)."""
        # Simple implementation - can be enhanced with more sophisticated algorithms
        # Remove whitespace and compare
        t1 = re.sub(r'\s+', '', text1.lower())
        t2 = re.sub(r'\s+', '', text2.lower())
        
        # If one is contained in the other
        if t1 in t2 or t2 in t1:
            return True
        
        # Calculate similarity ratio
        if len(t1) > 10 and len(t2) > 10:  # Only for longer strings
            # Simple character-based similarity
            shorter = t1 if len(t1) <= len(t2) else t2
            longer = t2 if len(t1) <= len(t2) else t1
            
            # Count matching characters
            matches = sum(1 for c in shorter if c in longer)
            similarity = matches / len(shorter)
            
            return similarity > 0.8  # 80% similarity threshold
        
        return False
    
    def _detect_columns(self, page, text_blocks, simplified=False):
        """Detect if the page has multiple columns and identify them.
        
        Args:
            page: The PDF page object
            text_blocks: List of text blocks with their bounding boxes
            simplified: If True, use a faster algorithm with fewer features
        """
        if not text_blocks:
            return []
        
        # Get page dimensions
        page_width = float(page.width)
        
        if simplified:
            # Fast column detection - check for common 2-column or 3-column patterns
            # This is much faster but less accurate for complex layouts
            
            # Check if we have a single text block (from simplified extraction)
            if len(text_blocks) == 1 and text_blocks[0].get('width', 0) > page_width * 0.8:
                # If we have a single block covering most of the page width,
                # we can't reliably detect columns in simplified mode
                return [text_blocks]
            
            # Simple heuristic: check if text blocks form clear left/right pattern
            left_blocks = []
            right_blocks = []
            middle_blocks = []
            
            for block in text_blocks:
                center_x = block.get('x_center', block['bbox'][0] + (block['bbox'][2] - block['bbox'][0])/2)
                if center_x < page_width * 0.33:  # Left third
                    left_blocks.append(block)
                elif center_x > page_width * 0.67:  # Right third
                    right_blocks.append(block)
                else:  # Middle third
                    middle_blocks.append(block)
            
            # Check for 2-column layout
            if len(left_blocks) > 2 and len(right_blocks) > 2 and len(middle_blocks) < len(left_blocks) + len(right_blocks):
                return [left_blocks, right_blocks]
            
            # Check for 3-column layout
            if len(left_blocks) > 1 and len(middle_blocks) > 1 and len(right_blocks) > 1:
                return [left_blocks, middle_blocks, right_blocks]
            
            # Default to single column in simplified mode
            return [text_blocks]
        
        # Full detailed column detection follows
        # Extract x-positions of text blocks
        x_positions = [(block['bbox'][0], block['bbox'][2]) for block in text_blocks]
        x_starts = [pos[0] for pos in x_positions]
        
        # Sort and look for natural breaks in x-positions
        sorted_starts = sorted(x_starts)
        gaps = [sorted_starts[i+1] - sorted_starts[i] for i in range(len(sorted_starts)-1)]
        
        if not gaps:
            return [text_blocks]  # Single column
        
        # Find significant gaps (indicating column separation)
        avg_gap = sum(gaps) / len(gaps)
        std_dev = (sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5
        significant_gap = avg_gap + std_dev * 1.5
        
        # Find potential column boundaries
        column_boundaries = [0]  # Start with left edge
        
        for i, gap in enumerate(gaps):
            if gap > significant_gap and sorted_starts[i+1] - sorted_starts[0] > page_width * 0.15:
                # This is a significant gap and we're at least 15% into the page
                column_boundaries.append((sorted_starts[i] + sorted_starts[i+1]) / 2)  # Use midpoint
        
        column_boundaries.append(page_width)  # End with right edge
        
        # If we found multiple columns
        if len(column_boundaries) > 2:
            # Group text blocks into columns
            columns = [[] for _ in range(len(column_boundaries)-1)]
            
            for block in text_blocks:
                x0 = block['bbox'][0]
                # Find which column this block belongs to
                for i in range(len(column_boundaries)-1):
                    if x0 >= column_boundaries[i] and x0 < column_boundaries[i+1]:
                        columns[i].append(block)
                        break
            
            # Remove empty columns
            columns = [col for col in columns if col]
            
            # If we have multiple non-empty columns
            if len(columns) > 1:
                return columns
        
        return [text_blocks]  # Default to single column
    
    def _extract_text_from_columns(self, page, text_blocks):
        """Extract text from a page with columns in the correct reading order."""
        columns = self._detect_columns(page, text_blocks)
        
        if len(columns) <= 1:
            # If no clear columns, fall back to normal extraction
            return page.extract_text()
        
        # Process each column in left-to-right order
        column_texts = []
        for column in columns:
            # Sort blocks in this column by their vertical position (top to bottom)
            sorted_blocks = sorted(column, key=lambda b: b['bbox'][1])
            column_text = '\n'.join(block['text'] for block in sorted_blocks)
            column_texts.append(column_text)
        
        # Join columns, separated by extra newlines
        return '\n\n'.join(column_texts)
    
    def _detect_toc_pages(self, pdf):
        """Detect pages that appear to be a table of contents."""
        toc_pages = []
        
        for i, page in enumerate(pdf.pages[:min(10, len(pdf.pages))]):
            # Get text from the page
            text = page.extract_text()
            if not text:
                continue
            
            # Check for TOC indicators
            toc_indicators = [
                r'\bcontents\b',
                r'\btable of contents\b',
                r'\bindex\b',
                r'\btoc\b'
            ]
            
            # Look for patterns of numbers followed by dots (like "1.....10")
            page_number_pattern = r'\b\d+[\s\.]+\d+\b'
            
            # Count TOC-like lines (chapter/section followed by page number)
            lines = text.lower().split('\n')
            toc_like_lines = 0
            
            for line in lines:
                if re.search(page_number_pattern, line):
                    toc_like_lines += 1
            
            # Check if this page looks like a TOC
            if any(re.search(pattern, text.lower()) for pattern in toc_indicators) or toc_like_lines > 5:
                toc_pages.append(i)
        
        return toc_pages
