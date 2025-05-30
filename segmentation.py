import re  # Regular expressions for pattern matching
import logging  # Logging for diagnostics
import nltk  # Natural language toolkit for text analysis
import json  # JSON handling for structured data
import string  # String utilities
from collections import defaultdict  # For counting and categorizing
from nltk.tokenize import sent_tokenize  # Sentence boundary detection
from datetime import datetime  # Date handling and validation
import difflib  # For fuzzy text matching

#==========================================================================
# Document Segmenter - Handles hierarchical segmentation of document text
#==========================================================================

class DocumentSegmenter:
    """Analyzes document structure and segments text into a hierarchical structure.
    
    This class handles the core document structure analysis, identifying sections,
    subsections, and their hierarchical relationships. It extracts headings,
    determines hierarchy levels, and organizes content into logical segments.
    
    Key capabilities:
    - Identify headings and subheadings using pattern and feature analysis
    - Build hierarchical document structure with proper nesting
    - Extract metadata like dates and sources from segments
    - Handle various document formats (reports, articles, contracts, etc.)
    """
    
    def __init__(self, use_machine_learning=False):
        """Initialize the document segmenter.
        
        Args:
            use_machine_learning: Whether to use ML-based heading detection.
                Currently experimental - rule-based approach performs better
                on most business documents.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.segment_cache = {}  # Cache results to avoid reprocessing identical documents
        self.use_machine_learning = use_machine_learning  # Future capability flag
        
        # Configuration for heading detection
        # These parameters were determined through empirical testing on
        # hundreds of business documents to maximize accuracy
        self.heading_config = {
            'min_confidence': 0.6,  # Minimum confidence score to consider a line a heading
            'max_heading_words': 15,  # Most headings are short; longer ones are likely paragraphs
            'strong_prefixes': [
                # Common document section indicators that strongly suggest a heading
                'chapter', 'section', 'part', 'appendix', 'exhibit',
                'schedule', 'addendum', 'annex'
            ],
            # Font characteristics typical of headings
            'heading_font_characteristics': {
                'bold': 2.0,       # Weight multiplier if bold
                'larger': 1.5,      # Weight multiplier if larger than body text
                'uppercase': 1.3,    # Weight multiplier if ALL CAPS
                'centered': 1.2,     # Weight multiplier if centered
                'numbered': 1.5      # Weight multiplier if it has a section number
            }
        }
        
        # Download NLTK resources if not already available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Define patterns for section headings with different numbering schemes
        self.heading_patterns = [
            # Numeric patterns
            r'^(\d+)\s+([A-Z].*?)$',                  # "1 Introduction"
            r'^(\d+\.\d+)\s+([A-Z].*?)$',             # "1.1 Background"
            r'^(\d+\.\d+\.\d+)\s+([A-Z].*?)$',        # "1.1.1 Details"
            r'^(\d+\.\d+\.\d+\.\d+)\s+([A-Z].*?)$',   # "1.1.1.1 Sub-details"
            
            # Roman numeral patterns
            r'^([IVXLCDM]+)\s+([A-Z].*?)$',           # "I Introduction"
            r'^([IVXLCDM]+\.[IVXLCDM]+)\s+([A-Z].*?)$', # "I.II Background"
            
            # Alphabetic patterns
            r'^([A-Z])\s+([A-Z].*?)$',                # "A Introduction"
            r'^([A-Z]\.[A-Z])\s+([A-Z].*?)$',          # "A.B Background"
            
            # Common heading patterns
            r'^(CHAPTER \d+:?)\s+(.+)$',              # "CHAPTER 1: Title"
            r'^(SECTION \d+:?)\s+(.+)$',              # "SECTION 1: Title"
            r'^(ARTICLE \d+:?)\s+(.+)$',              # "ARTICLE 1: Title"
            r'^(PART \d+:?)\s+(.+)$',                 # "PART 1: Title"
            
            # Special formats
            r'^(§\s*\d+(\.\d+)*)\s+(.+)$',           # "§ 1.2 Title"
            r'^\*\*\*\s*(.+?)\s*\*\*\*$',              # "*** Title ***"
            r'^={3,}\s*(.+?)\s*={3,}$'                 # "==== Title ===="
        ]
    
    def segment_document(self, text, layout_info=None, page_metadata=None):
        """Segment document text into a hierarchical structure of logical sections.
        
        Main entry point for document segmentation. Performs the following steps:
        1. Extract and analyze potential headings in the document
        2. Determine hierarchy levels and relationships
        3. Build segments with proper nesting and hierarchy
        4. Extract metadata (dates, sources) for each segment
        
        Args:
            text: Full document text to segment
            layout_info: Optional layout information from document reader
            page_metadata: Optional page mapping information for cross-referencing
            
        Returns:
            List of segment dictionaries with hierarchical structure
        """
        self.logger.info("Starting optimized document segmentation")
        
        # Input validation - empty or non-string inputs aren't processable
        if not text or not isinstance(text, str):
            self.logger.warning("Invalid text input, returning empty segments")
            return []
            
        # Performance optimization: Check result cache first
        # We use a composite hash of the first and last sections of the document
        # instead of hashing the entire text for better performance.
        # This approach catches most duplicate documents while being much faster.
        text_hash = hash(text[:min(500, len(text))] + text[-min(500, len(text)):])  
        if text_hash in self.segment_cache:
            self.logger.info("Using cached segmentation results - document already processed")
            return self.segment_cache[text_hash]
        
        # Quick check if document is too short for complex segmentation
        if len(text) < 1000:
            self.logger.info("Document too short for segmentation, creating single segment")
            # Create a meaningful title from the beginning of the document
            first_line = text.split('\n', 1)[0].strip() if text else ''
            doc_title = first_line[:50] if len(first_line) < 50 else "Document"
            
            segment = {
                "segment_level": 1,
                "segment_title": doc_title,
                "segment_text": text,
                "segment_date": self._extract_date(text),
                "segment_source": self._extract_source(text, 0),
                "start_index": 0,
                "end_index": len(text),
                "named_entities": {"persons": [], "organizations": [], "locations": [], "dates": [], "misc": []}
            }
            return [segment]
        
        # Step 1: Extract potential headings with optimized algorithm
        self.logger.info("Extracting headings with optimized algorithm")
        potential_headings = self._extract_potential_headings(text)
        
        # Ensure we have reasonable number of headings for best performance
        if len(potential_headings) > 80:  # Increased limit for more complex documents
            self.logger.info(f"Limiting from {len(potential_headings)} to 80 headings for performance")
            # Sort by confidence and keep only top headings
            potential_headings.sort(key=lambda h: h.get('confidence', 0), reverse=True)
            potential_headings = potential_headings[:80]
        
        # Step 2: Quick refinement of heading levels
        self.logger.info("Quick heading level refinement")
        self._refine_heading_levels(potential_headings)
        
        # Step 3: Document structure analysis
        self.logger.info("Document structure analysis")
        doc_structure = self._analyze_document_structure(text, potential_headings, layout_info)
        
        # Step 4: Build segments with enhanced algorithm
        self.logger.info("Building segments with enhanced algorithm")
        segments = self._build_hierarchical_segments(text, doc_structure, page_metadata)
        
        # Step 5: Ensure each segment has the required structure
        if segments:
            for segment in segments:
                # Ensure segment has all required fields
                if "named_entities" not in segment:
                    segment["named_entities"] = {
                        "persons": [], "organizations": [], "locations": [],
                        "dates": [], "misc": []
                    }
                    
                # Make sure all necessary entity types are present
                for entity_type in ["persons", "organizations", "locations", "dates", "misc"]:
                    if entity_type not in segment["named_entities"]:
                        segment["named_entities"][entity_type] = []
        
        self.segment_cache[text_hash] = segments
        
        self.logger.info(f"Optimized segmentation complete: {len(segments)} segments identified")
        return segments
        
    def _extract_potential_headings(self, text):
        self.logger.debug("Fast extraction of potential headings")
        potential_headings = []
        
        # Early exit for very short documents - optimization
        if len(text) < 500:
            return []
            
        # Use faster line splitting with max line limit for performance
        lines = text.split('\n', 5000)  # Limit to first 5000 lines for very large documents
        
        # Skip full statistics calculation for large documents
        max_lines_for_stats = 1000  # Even faster with smaller sample
        sample_lines = lines[:max_lines_for_stats] if len(lines) > max_lines_for_stats else lines
        
        # Calculate statistics from sample for better performance
        line_lengths = [len(line.strip()) for line in sample_lines if line.strip() and len(line.strip()) < 200]
        avg_line_length = sum(line_lengths) / max(1, len(line_lengths)) if line_lengths else 60  # Default if no data
        
        # Pre-compile frequent regex patterns for performance
        numeric_prefix_pattern = re.compile(r'^\d+(\.\d+)*$')
        numeric_prefix_with_space = re.compile(r'^\d+(\.\d+)*\s')
        
        # Use a faster line position tracking approach
        line_positions = {}
        cumulative_length = 0
        
        for i, line in enumerate(lines):
            line_length = len(line)
            line_positions[i] = cumulative_length
            cumulative_length += line_length + 1  # +1 for the newline character
            
            # Skip empty lines and very long lines (unlikely headings)
            line = line.strip()
            if not line or len(line) > 120 or len(line) < 3:
                continue
            
            spaces = line.count(' ')
            if spaces > self.heading_config['max_heading_words']:
                continue
            
            confidence = 0.0
            heading_text = ""
            heading_level = 0
            prefix = ""
            if line.startswith(('Chapter ', 'CHAPTER ', 'Section ', 'SECTION ')):
                heading_text = line
                heading_level = 1
                confidence = 0.85
                if line.startswith(('Chapter ', 'CHAPTER ')):
                    prefix = line.split(' ')[0]
                    heading_text = ' '.join(line.split(' ')[1:]).strip()
            
            # Check for ALL CAPS - very likely headings
            elif line.isupper() and len(line) < 100:
                heading_text = line
                heading_level = 1
                confidence = 0.75
                
            # Check for numeric prefix patterns (1.2.3) - common in structured documents
            elif numeric_prefix_with_space.match(line):
                parts = line.split(' ', 1)
                prefix = parts[0]
                heading_text = parts[1] if len(parts) > 1 else ''
                heading_level = prefix.count('.') + 1
                confidence = 0.85
                
            # Selective pattern matching - only if line passes initial filters
            elif spaces < 10 and len(line) < avg_line_length * 0.7:
                # Only check the most common/reliable patterns for speed
                for pattern_idx, pattern in enumerate(self.heading_patterns[:2]):  # Use only 2 most important patterns
                    match = re.match(pattern, line)
                    if match:
                        pattern_confidence = 0.75
                        
                        if len(match.groups()) >= 2:
                            prefix = match.group(1)
                            heading_text = match.group(2) if len(match.groups()) > 1 else line
                            
                            if numeric_prefix_pattern.match(prefix):
                                heading_level = prefix.count('.') + 1
                            else:
                                heading_level = 1
                        else:
                            heading_text = match.group(1) if match.groups() else line
                            heading_level = 1
                        
                        confidence = pattern_confidence
                        break
                
                # Quick heuristic check for likely headings (faster check)
                if confidence < 0.5 and len(line) < avg_line_length * 0.5:
                    if line[0].isupper() and not line[-1] in '.?!,:;':
                        confidence = 0.6
                        heading_text = line
                        heading_level = 1
            
            # If we have enough confidence, add as heading
            if confidence >= self.heading_config['min_confidence']:
                # Use pre-calculated positions for speed
                line_position = line_positions[i]
                
                # Add to potential headings with calculated position
                potential_headings.append({
                    "text": heading_text.strip() or line,
                    "prefix": prefix,
                    "level": heading_level,
                    "position": i,
                    "start_index": line_position + (len(line) - len(line.lstrip())),
                    "end_index": line_position + len(line.rstrip()),
                    "confidence": confidence
                })
                
                # Early termination with enough headings - avoids processing entire document
                if len(potential_headings) >= 100:  # Cap at reasonable number
                    self.logger.info("Fast heading extraction: reached heading limit (100)")
                    break
        
        # Sort headings by position (usually already in order, but ensure it)
        potential_headings.sort(key=lambda h: h["start_index"])
        return potential_headings
        
        # Post-process to improve level assignments based on sequence
        self._refine_heading_levels(potential_headings)
        
        self.logger.debug(f"Extracted {len(potential_headings)} potential headings")
        return potential_headings
    
    def _refine_heading_levels(self, headings):
        if not headings:
            return
        
        # Group headings by similar prefixes
        prefix_groups = defaultdict(list)
        for i, heading in enumerate(headings):
            prefix = heading.get("prefix", "")
            if prefix:
                # Simplify numeric prefixes to their pattern (1.2.3 -> numeric.3)
                if re.match(r'^\d+(\.\d+)*$', prefix):
                    prefix_pattern = f"numeric.{prefix.count('.')+1}"
                    prefix_groups[prefix_pattern].append(i)
                # Group common textual prefixes
                elif any(prefix.lower().startswith(p) for p in self.heading_config['strong_prefixes']):
                    for p in self.heading_config['strong_prefixes']:
                        if prefix.lower().startswith(p):
                            prefix_groups[p].append(i)
                            break
                else:
                    # Other patterns
                    prefix_groups["other"].append(i)
        
        # Process each prefix group to ensure consistent levels
        for prefix, indices in prefix_groups.items():
            if prefix.startswith("numeric."):
                # For numeric prefixes, level is already embedded in the pattern
                level = int(prefix.split('.')[-1])
                for idx in indices:
                    headings[idx]["level"] = level
            elif prefix in self.heading_config['strong_prefixes']:
                # Common prefixes (Chapter, Section, etc.) are usually top-level
                for idx in indices:
                    headings[idx]["level"] = 1
        
        # Final pass: ensure heading levels make hierarchical sense
        current_levels = {1: None}  # Track the last heading at each level
        for i, heading in enumerate(headings):
            level = heading["level"]
            
            # Ensure levels don't skip (e.g., can't have level 3 without level 2)
            while level > 1 and level - 1 not in current_levels:
                level -= 1
            
            # Update the heading's level if needed
            headings[i]["level"] = level
            current_levels[level] = i
            
            # Clear lower levels when a higher level is encountered
            levels_to_remove = [l for l in current_levels if l > level]
            for l in levels_to_remove:
                current_levels.pop(l, None)
    
    def _analyze_document_structure(self, text, potential_headings, layout_info=None):
        self.logger.debug("Fast document structure analysis")
        
        # Initialize document structure with minimal processing
        doc_structure = {
            "headings": potential_headings,
            "hierarchy_type": "unknown",
            "has_toc": False,
            "estimated_segments": len(potential_headings)
        }
        
        # Fast path for empty headings
        if not potential_headings:
            doc_structure["hierarchy_type"] = "flat"
            return doc_structure
            
        # Quick hierarchy type determination
        # Instead of sorting the entire list, just check key attributes
        if len(potential_headings) < 5:
            # For very few headings, treat as flat for speed
            doc_structure["hierarchy_type"] = "flat"
        else:
            # Quick check for top-level headings count
            level_1_count = sum(1 for h in potential_headings if h["level"] == 1)
            
            # If at least 80% are level 1, it's likely flat
            if level_1_count >= 0.8 * len(potential_headings):
                doc_structure["hierarchy_type"] = "flat"
            # If we have multiple levels, check if we have a good distribution
            elif any(h["level"] > 1 for h in potential_headings):
                # Skip the expensive sorted() == range() check and use a simpler heuristic
                has_level_2 = any(h["level"] == 2 for h in potential_headings)
                has_level_3 = any(h["level"] == 3 for h in potential_headings)
                
                if has_level_2 and has_level_3:
                    doc_structure["hierarchy_type"] = "hierarchical"
                else:
                    doc_structure["hierarchy_type"] = "partial_hierarchical"
            else:
                doc_structure["hierarchy_type"] = "flat"
        
        # Skip TOC detection for performance unless layout_info already has it
        if layout_info and layout_info.get('toc_pages'):
            doc_structure["has_toc"] = True
        elif len(text) < 20000:  # Only check for TOC in smaller documents
            # Only check first 1000 chars for extremely fast TOC detection
            first_portion = text[:1000].lower()
            if "table of contents" in first_portion or "contents:" in first_portion:
                doc_structure["has_toc"] = True
        
        return doc_structure
    
    def _build_hierarchical_segments(self, text, doc_structure, page_metadata=None):
        segments = []
        headings = doc_structure.get('headings', [])
        
        # Return single segment if no headings found
        if not headings:
            # Create a meaningful title for untitled documents
            first_line = text.split('\n', 1)[0].strip() if text else ''
            doc_title = first_line[:50] if len(first_line) < 50 else "Document"
            
            segment = {
                "segment_level": 1,
                "segment_title": doc_title,
                "segment_text": text,
                "segment_date": self._extract_date(text[:1000]),
                "segment_source": self._extract_source(text, 0),
                "start_index": 0,
                "end_index": len(text),
                "pages": self._get_pages_for_segment(0, len(text), page_metadata) if page_metadata else []
            }
            return [segment]
        
        # Prepare and sort headings by their position in the document
        sorted_headings = sorted(headings, key=lambda h: h.get('start_index', 0))
        
        # Enhanced segment building algorithm
        for i, heading in enumerate(sorted_headings):
            # Get segment level (normally from heading but fallback to 1)
            level = heading.get('level', 1)
            title = heading.get('text', '').strip() or f"Section {i+1}"
            
            # Store numeric prefix if present (e.g., "8.8") for inclusion in segment data
            numeric_prefix = ""
            numeric_match = re.match(r'^(\d+(?:\.\d+)*)\s*[.:;\s]*\s*', title)
            if numeric_match:
                numeric_prefix = numeric_match.group(1)
            
            # Remove only text prefixes like "Chapter" or "Section" but keep numeric ones
            title_without_text_prefix = re.sub(r'^(chapter|section|part)\s+\d+[.:;\s]*', '', title, flags=re.IGNORECASE).strip()
            
            # Only remove numeric prefix from display title if we're storing it separately
            if numeric_prefix:
                # Keep the numeric prefix in the title but clean up spacing/punctuation
                clean_text = re.sub(r'^\d+(?:\.\d+)*\s*[.:;\s]*\s*', '', title_without_text_prefix).strip()
                title = f"{numeric_prefix} {clean_text}".strip()
            
            # Determine precise start index - include the heading itself
            start_idx = heading.get('start_index', 0)
            
            # Determine end index (either next heading or end of document)
            end_idx = len(text)
            if i < len(sorted_headings) - 1:
                end_idx = sorted_headings[i+1].get('start_index', len(text))
            
            # Extract segment text using precise indices
            segment_text = text[start_idx:end_idx].strip()
            
            # Skip minimal segments - any segment under 15 characters is likely a processing artifact
            if len(segment_text) < 15:
                continue
                
            # Find natural segment boundaries (prefer paragraph breaks)
            # This ensures text isn't cut off mid-paragraph
            if i < len(sorted_headings) - 1 and end_idx < len(text):
                # Look for paragraph breaks near the boundary
                next_start = sorted_headings[i+1].get('start_index', end_idx)
                # Adjust buffer size based on document size for better performance
                buffer_size = 50 if len(text) > 100000 else 150
                buffer_zone = text[max(0, next_start-buffer_size):min(len(text), next_start+5)]
                
                # Try to find a natural break point (paragraph or sentence ending)
                paragraph_breaks = [m.start() for m in re.finditer(r'\n\s*\n', buffer_zone)]
                if paragraph_breaks:
                    # Find the closest paragraph break to the heading
                    closest_break = max([b for b in paragraph_breaks if b <= buffer_size//2] or [0])
                    end_idx = max(0, next_start-buffer_size) + closest_break
                else:
                    # If no paragraph breaks, try sentence endings
                    sentence_breaks = [m.start() for m in re.finditer(r'[.!?]\s+', buffer_zone[:buffer_size//2])]
                    if sentence_breaks:
                        closest_break = max(sentence_breaks)
                        end_idx = max(0, next_start-buffer_size) + closest_break + 1  # Include the period
                
                # Update segment text with refined boundaries
                segment_text = text[start_idx:end_idx].strip()
            
            # Extract date and source from the segment
            segment_date = self._extract_date(segment_text[:min(len(segment_text), 500)])
            segment_source = self._extract_source(segment_text, start_idx)
            
            # Create segment with complete information
            segment = {
                "segment_level": level,
                "segment_title": title,
                "segment_text": segment_text,
                "segment_date": segment_date,
                "segment_source": segment_source,
                "start_index": start_idx,
                "end_index": end_idx,
            }
            
            # Add page mapping if metadata available
            if page_metadata:
                segment["pages"] = self._get_pages_for_segment(start_idx, end_idx, page_metadata)
            
            segments.append(segment)
        
        return segments
    def _get_pages_for_segment(self, start_index, end_index, page_metadata):
        segment_pages = []
        
        for page in page_metadata:
            # Check if this page overlaps with the segment
            if (start_index <= page["end_index"] and end_index >= page["start_index"]):
                segment_pages.append(page["page_number"])
        
        return segment_pages
    
    def _post_process_segments(self, text, segments):
        self.logger.debug("Post-processing segments")
        
        if not segments:
            return segments
            
        # Sort segments by start_index to ensure proper order
        segments.sort(key=lambda s: s["start_index"])
        
        # Ensure segments don't overlap incorrectly
        for i in range(len(segments) - 1):
            if segments[i]["end_index"] >= segments[i+1]["start_index"]:
                segments[i]["end_index"] = segments[i+1]["start_index"] - 1
                
                # Also update segment text
                segments[i]["segment_text"] = text[segments[i]["start_index"]:segments[i]["end_index"]+1].strip()
        
        # Handle segment nesting based on levels
        # This is a simplified approach - for a full implementation we would maintain
        # parent-child relationships and adjust start/end indices accordingly
        current_parent_by_level = {1: None}  # Track the current parent at each level
        
        for i, segment in enumerate(segments):
            level = segment["segment_level"]
            
            # Check for missing levels (e.g., level 3 without level 2)
            parent_level = level - 1
            while parent_level > 0 and parent_level not in current_parent_by_level:
                parent_level -= 1
            
            # Record this segment as the current segment at its level
            current_parent_by_level[level] = i
            
            # Clear higher levels when lower level is encountered
            levels_to_remove = [l for l in current_parent_by_level.keys() if l > level]
            for l in levels_to_remove:
                current_parent_by_level.pop(l, None)
        
        return segments
    
    def _extract_segment_metadata(self, text, segments):
        self.logger.debug("Extracting segment metadata")
        
        for segment in segments:
            # Extract date information
            segment["segment_date"] = self._extract_date(segment["segment_text"])
            
            # If no date in segment text, check segment title
            if not segment["segment_date"] and segment["segment_title"]:
                segment["segment_date"] = self._extract_date(segment["segment_title"])
            
            # Extract source information
            segment["segment_source"] = self._extract_source(segment["segment_text"], segment["start_index"])
            
            # Clean and validate the extracted data
            if segment["segment_date"] and not self._validate_date(segment["segment_date"]):
                self.logger.debug(f"Invalid date format: {segment['segment_date']}")
                segment["segment_date"] = None
        
        return segments

    def _extract_date(self, text):
        if not text:
            return None
            
        # Cache common date patterns using compiled regex for speed
        if not hasattr(self, '_compiled_date_patterns'):
            self._compiled_date_patterns = [
                # ISO format: YYYY-MM-DD (high priority)
                re.compile(r'\b(\d{4}-\d{1,2}-\d{1,2})\b'),
                
                # American format: MM/DD/YYYY
                re.compile(r'\b(\d{1,2}/\d{1,2}/\d{4})\b'),
                re.compile(r'\b(\d{1,2}-\d{1,2}-\d{4})\b'),
                
                # European format: DD/MM/YYYY
                re.compile(r'\b(\d{1,2}\.\d{1,2}\.\d{4})\b'),
                
                # Written formats with explicit labels (highest priority)
                re.compile(r'\b(?:Date|Dated|As of date|Publication date|Effective date|Issue date)\s*:\s*(\S.*?\d{4})\b', re.IGNORECASE),
                re.compile(r'\b(?:Revision|Updated|Published|Released)\s*(?:date|on)?\s*:\s*(\S.*?\d{4})\b', re.IGNORECASE),
                
                # Written formats
                re.compile(r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})\b', re.IGNORECASE),
                re.compile(r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec),?\s+\d{4})\b', re.IGNORECASE),
                re.compile(r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\b', re.IGNORECASE),
                
                # Quarters and fiscal years
                re.compile(r'\b(Q[1-4]\s+\d{4})\b'),
                re.compile(r'\b(FY\s*\d{4})\b'),
                re.compile(r'\b(Fiscal\s+Year\s+\d{4})\b'),
                
                # Year only (lowest priority)
                re.compile(r'\b(\d{4})\b')
            ]
        
        # Look for dates using patterns in priority order
        for pattern in self._compiled_date_patterns:
            matches = pattern.findall(text)
            if matches:
                # Take the first match (most likely to be relevant)
                return matches[0].strip()
        
        # If no specific date pattern matched, check for just a year
        year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
        if year_matches:
            # Count occurrences of each year
            year_counts = {}
            for year in year_matches:
                if year in year_counts:
                    year_counts[year] += 1
                else:
                    year_counts[year] = 1
            
            # Find the most frequent year (likely to be the document date)
            most_common_year = max(year_counts.items(), key=lambda x: x[1])[0]
            return most_common_year
            
        return None
    
    def _validate_date(self, date_str):
        if not date_str:
            return False
            
        # Simple validation: check if the string contains a year (19xx or 20xx)
        if re.search(r'(19|20)\d{2}', date_str):
            # Additional checks could be added for month/day validity
            return True
            
        return False
    
    def _extract_source(self, text, position):
        if not text:
            return None
            
        # Look within the first or last paragraphs (where source info often appears)
        paragraphs = text.split('\n\n')
        search_texts = []
        
        if paragraphs:
            # Check the first and last two paragraphs (if they exist)
            if len(paragraphs) >= 1:
                search_texts.append(paragraphs[0])  # First paragraph
            if len(paragraphs) >= 2:
                search_texts.append(paragraphs[-1])  # Last paragraph
            if len(paragraphs) >= 3:
                search_texts.append(paragraphs[1])  # Second paragraph
            if len(paragraphs) >= 4:
                search_texts.append(paragraphs[-2])  # Second-to-last paragraph
        else:
            search_texts = [text]
        
        # Source patterns (ordered by likelihood/specificity)
        source_patterns = [
            # Explicit source markers
            r'Source\s*:?\s+([^\n\.]+(?:\.[^\n\.]+){0,5})',
            r'From\s*:?\s+([^\n\.]+(?:\.[^\n\.]+){0,3})',
            r'By\s*:?\s+([^\n\.]+(?:\.[^\n\.]+){0,3})',
            r'Author(?:s|ed by)?\s*:?\s+([^\n\.]+(?:\.[^\n\.]+){0,3})',
            r'(?:Prepared|Written|Compiled)\s+by\s*:?\s+([^\n\.]+(?:\.[^\n\.]+){0,3})',
            r'Courtesy\s+of\s*:?\s+([^\n\.]+(?:\.[^\n\.]+){0,3})',
            
            # Copyright markers
            r'(?:©|Copyright)\s*(?:[Cc])?\s*(\d{4})?\s*(?:by)?\s+([^\n\.]+(?:\.[^\n\.]+){0,3})',
            
            # Publication info
            r'(?:Published|Issued)\s+by\s+([^\n\.]+(?:\.[^\n\.]+){0,3})',
            
            # Institution/organization markers
            r'((?:[A-Z][a-z]*\s*){1,5}(?:University|Institute|Corporation|Inc\.|LLC|Ltd\.|Association|Organization|Department|Agency))',
            
            # Person attribution markers
            r'\*\s*([A-Z][a-z]+\s+[A-Z][a-z]+)\s*(?:is|was|works|serves)'
        ]
        
        # Search for source patterns in priority order in our search texts
        for search_text in search_texts:
            for pattern in source_patterns:
                matches = re.findall(pattern, search_text, re.IGNORECASE | re.DOTALL)
                if matches:
                    # Clean up the match
                    source = matches[0]
                    if isinstance(source, tuple):
                        # Some patterns might have multiple capture groups
                        source = ' '.join(s for s in source if s).strip()
                    
                    # Clean up the source text
                    source = re.sub(r'\s+', ' ', source).strip()
                    source = re.sub(r'^[,\.:\s]+|[,\.:\s]+$', '', source).strip()
                    
                    return source
        
        # Additional method: look for institutional affiliations
        institutional_match = re.search(r'\n([^\n]+(?:University|Institute|College|Laboratory|Foundation|Center))\s*\n', text, re.IGNORECASE)
        if institutional_match:
            return institutional_match.group(1).strip()
                
        return None
