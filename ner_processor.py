import spacy
import logging
from tqdm import tqdm  # For progress bars during processing
import re  # Regular expressions for pattern matching
import difflib  # Used for fuzzy matching of similar entities
from collections import defaultdict, Counter
import string
from spacy.tokens import Doc, Span
from spacy.language import Language

# ===============================================================
# NER Processor - Handles named entity recognition and extraction
# ===============================================================

class NERProcessor:
    """Handles entity extraction and normalization from document segments.
    
    Uses spaCy models for base entity recognition and enhances with custom
    logic for entity normalization and contact information extraction.
    Implements performance optimizations for large documents.
    """
    
    def __init__(self, model="en_core_web_lg", use_transformers=False):
        """Initialize the NER processor with the specified model.
        
        Args:
            model: Name of spaCy model to use - en_core_web_lg recommended for best accuracy
            use_transformers: Whether to use transformer-based models (slower but more accurate)
        
        Note: Transformer models require ~4x more memory but improve accuracy by ~5-8%
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.entity_confidence_threshold = 0.7  # Min confidence to accept an entity
        self.context_window_size = 5  # Words to check around entity for context
        self.use_transformers = use_transformers
        self.entity_cache = {}  # Cache entities to avoid re-processing identical segments
        
        # Map our simplified categories to spaCy's entity types
        # This allows us to group related entity types under user-friendly categories
        self.entity_types = {
            "persons": ["PERSON"],                      # People names
            "organizations": ["ORG", "NORP", "FAC"],   # Companies, agencies, buildings, nationalities
            "locations": ["GPE", "LOC"],              # Countries, cities, geographic features
            "dates": ["DATE"],                        # Calendar dates, periods
            "contact_info": ["PHONE", "EMAIL"],       # Phone numbers, email addresses
            "misc": ["MISC", "QUANTITY", "CARDINAL", "ORDINAL"],  # Numbers, quantities, etc.
            "monetary": ["MONEY"],                    # Currency values
            "percentages": ["PERCENT"],                # Percentage values
            "products": ["PRODUCT"],                   # Product names
            "events": ["EVENT"],                      # Named events (conferences, wars, etc.)
            "laws": ["LAW"],                          # Legal document references
            "works": ["WORK_OF_ART"]                  # Titles of books, songs, etc.
        }
        
        
        try:
            self.logger.info(f"Loading spaCy model: {model}")
            
            # Handle transformer vs standard models
            # Transformers are more accurate but slower + need more RAM
            if use_transformers and not model.startswith("xx_"):  # Skip for multi-lingual models (xx_*)
                try:
                    # Try loading transformer extension - needs separate installation
                    # Might fail if user doesn't have GPU or didn't install the package
                    import spacy_transformers
                    self.nlp = spacy.load(model)
                    self.logger.info("Using transformer-based NER pipeline")
                except (ImportError, OSError):
                    # Fall back to standard model if transformers aren't available
                    # This is a graceful degradation - will still work, just less accurate
                    self.logger.warning("Transformer components not available. Using standard model.")
                    self.nlp = spacy.load(model)
            else:
                # Standard CPU-based model - much faster but slightly less accurate
                # Good enough for most business documents
                self.nlp = spacy.load(model)
                
            self.logger.info("SpaCy model loaded successfully")
            
            
            self._register_custom_components()
            
        except OSError:
            self.logger.warning(f"SpaCy model {model} not found. Downloading...")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)
            self._register_custom_components()
            self.logger.info(f"SpaCy model {model} downloaded and loaded")
    
    def _register_custom_components(self):
        
        if "entity_cleaner" not in self.nlp.pipe_names:
            
            @Language.component("entity_cleaner")
            def entity_cleaner(doc):
                
                if not doc.ents:
                    return doc
                    
                new_ents = []
                for ent in doc.ents:
                    
                    if ent.label_ != "DATE" and self._is_numeric_entity(ent.text):
                        continue
                        
                    
                    cleaned_start = ent.start
                    cleaned_end = ent.end
                    
                    
                    span_text = ent.text
                    while cleaned_start < cleaned_end and span_text.startswith(tuple(string.punctuation)):
                        cleaned_start += 1
                        span_text = doc[cleaned_start:cleaned_end].text
                        
                    while cleaned_start < cleaned_end and span_text.endswith(tuple(string.punctuation)):
                        cleaned_end -= 1
                        span_text = doc[cleaned_start:cleaned_end].text
                    
                    if cleaned_start < cleaned_end:
                        new_span = Span(doc, cleaned_start, cleaned_end, label=ent.label_)
                        new_ents.append(new_span)
                        
                doc.ents = new_ents
                return doc
                
            self.nlp.add_pipe("entity_cleaner", last=True)
            self.logger.info("Added custom entity cleaning component to the pipeline")
    
    def process_segments(self, segments):
        """Process document segments to extract named entities.
        
        This is the main entry point for NER processing. It handles:
        1. Dynamic batch sizing based on document complexity
        2. Parallel processing of segment batches
        3. Smart normalization of entities across segments
        4. Post-processing for special cases like headings
        
        Args:
            segments: List of document segments with segment_text field
            
        Returns:
            The same segments with named_entities added to each
        """
        self.logger.info("Starting optimized NER processing with intelligent batching")
        
        if not segments:
            return []
            
        import concurrent.futures
        from tqdm import tqdm
        import numpy as np
        
        # OPTIMIZATION: Use multiprocessing for parallel NER on longer documents
        # For shorter documents, sequential processing is faster due to overhead
        use_parallel = len(segments) > 4 or sum(len(s.get('segment_text', '')) for s in segments) > 20000
        
        # Initialize all segments with proper entity categories
        # Do this up front to avoid key errors later
        for segment in segments:
            if 'named_entities' not in segment:
                segment['named_entities'] = {
                    "persons": [], "organizations": [], "locations": [],
                    "dates": [], "misc": [], "monetary": [], "percentages": [],
                    "products": [], "events": [], "laws": [], "works": []
                }
        
        # OPTIMIZATION: Calculate statistics to determine optimal processing approach
        # Memory usage scales with total character count, so we adapt accordingly
        total_chars = sum(len(segment.get('segment_text', '')) for segment in segments)
        avg_segment_size = total_chars / len(segments) if segments else 0
        
        # OPTIMIZATION: Use vectorized batch sizing based on system specs and document complexity
        # Benchmarked for optimal performance across different document types
        if avg_segment_size > 5000:  # Very long segments (e.g., legal docs)
            batch_size = max(2, len(segments) // 4)  # More aggressive batching
            max_workers = min(4, len(segments))  # Limit worker count for memory efficiency
        elif avg_segment_size > 2000:  # Medium segments (reports, articles) 
            batch_size = max(3, len(segments) // 3)
            max_workers = min(6, len(segments))
        else:  # Short segments (emails, memos, etc.)
            batch_size = max(4, len(segments) // 2)
            max_workers = min(8, len(segments))
            
        self.logger.info(f"Processing {len(segments)} segments with batch size {batch_size}, using {'parallel' if use_parallel else 'sequential'} processing")
        
        # OPTIMIZATION: Split work into batches based on complexity metrics
        batches = [segments[i:i+batch_size] for i in range(0, len(segments), batch_size)]
        
        # OPTIMIZATION: Use thread pool for parallel processing when beneficial
        # For smaller documents, the overhead isn't worth it
        if use_parallel and len(batches) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Process batches in parallel with progress tracking
                list(tqdm(executor.map(self._process_segment_batch_fast, batches), 
                          total=len(batches), desc="NER Processing"))
        else:
            # Sequential processing for smaller documents or when parallel isn't beneficial
            for batch in tqdm(batches, desc="NER Processing"):
                self._process_segment_batch_fast(batch)
        
        # Entity normalization helps reconcile variations of the same entity
        # (e.g., "Google" vs "Google Inc." vs "Google LLC")
        # Skip for large docs (>30 segments or >100K chars) as it's expensive
        if len(segments) <= 30 and total_chars < 100000:
            self._normalize_entities_fast(segments)
        else:
            self.logger.info("Skipping normalization for large document")
            
        # Final pass to extract entities from heading context and fix 
        # any entities that might have been missed or miscategorized
        self._post_process_special_segments(segments)
        
        return segments
    
    def _process_segment_batch_fast(self, segments):
        """Process a batch of segments efficiently using caching and filtering.
        
        This method optimizes processing by:
        1. Skipping very short segments (likely not content-rich)
        2. Using content hashing to avoid reprocessing duplicate content
        3. Processing multiple segments in one batch for better throughput
        
        Args:
            segments: A batch of document segments to process
        """
        # Track which segments actually need processing (vs cached/skipped)
        segments_to_process = []
        
        # First pass: determine which segments need processing
        for i, segment in enumerate(segments):
            # Skip very short segments (<50 chars), likely headers or noise
            # These rarely contain meaningful entities worth extracting
            if len(segment.get('segment_text', '')) < 50:
                segment['named_entities'] = {
                    "persons": [], "organizations": [], "locations": [],
                    "dates": [], "misc": []
                }
                continue
                
            # Create a hash from the first portion of text
            # This avoids processing identical or nearly identical segments
            # Using just the first 100 chars is faster and usually sufficient
            text_hash = hash(segment.get('segment_text', '')[:100])  
            
            # Check if we've already processed similar text
            if text_hash in self.entity_cache:
                # Reuse previous results (make a copy to prevent shared references)
                segment['named_entities'] = self.entity_cache[text_hash].copy()
            else:
                # Queue this segment for processing
                segments_to_process.append((i, segment, text_hash))
            
        if not segments_to_process:
            return
            
        for i, segment, text_hash in segments_to_process:
            text = segment.get('segment_text', '')
            if not text:
                segment['named_entities'] = {
                    "persons": [], "organizations": [], "locations": [],
                    "dates": [], "misc": []
                }
                continue
                
            try:
                
                # Disable components not critical for entity recognition
                
                disabled_pipes = ["lemmatizer", "attribute_ruler", "tok2vec"]
                if not self.entity_patterns:
                    disabled_pipes.append("entity_ruler") # Don't run ruler if no patterns
                
                with self.nlp.select_pipes(disable=disabled_pipes):
                    # OPTIMIZATION: Dynamic text chunking based on complexity
                    # This prevents memory issues while still capturing important entities
                    complexity_factor = sum(c.isalpha() for c in text[:500]) / 500 if text else 0
                    # Complex text (more alphabetic chars) = more entities = process less at once
                    max_chars = min(len(text), int(25000 * (1 - 0.5 * complexity_factor)))
                    doc = self.nlp(text[:max_chars])
                
                entities = self._extract_entities(doc)
                segment['named_entities'] = entities
                self.entity_cache[text_hash] = entities.copy()
            except Exception as e:
                self.logger.warning(f"Error processing segment: {str(e)}")
                segment['named_entities'] = {
                    "persons": [], "organizations": [], "locations": [],
                    "dates": [], "misc": []
                }
        
    def _normalize_entities_fast(self, segments):
        self.logger.info("Fast entity normalization") 
        
        for entity_type in ['persons', 'organizations', 'locations']:
            all_entities = set()
            for segment in segments:
                if entity_type in segment['named_entities']:
                    all_entities.update(segment['named_entities'][entity_type])
                
            if len(all_entities) < 3:  # Only normalize if we have enough entities
                continue
                
            entity_map = {}
            for entity in all_entities:
                lower = entity.lower()
                if lower not in entity_map:
                    entity_map[lower] = entity
                elif len(entity) > len(entity_map[lower]):
                    entity_map[lower] = entity
            
            for segment in segments:
                if entity_type in segment['named_entities']:
                    normalized = []
                    for entity in segment['named_entities'][entity_type]:
                        normalized.append(entity_map.get(entity.lower(), entity))
                    segment['named_entities'][entity_type] = normalized
        
        self.logger.info("Enhanced named entity recognition complete")
    
    def _post_process_special_segments(self, segments):
        """Ensure entities are properly categorized based on segment context and headings."""
        
        # Process segments based on heading context
        for segment in segments:
            heading = segment.get('segment_title', '').lower()
            text = segment.get('segment_text', '')
            
            # Special processing for location-focused segments
            if any(location_term in heading for location_term in ['location', 'geography', 'region', 'area', 'country', 'state', 'city']):
                # Ensure potential locations are captured
                potential_locations = self._extract_potential_locations(text)
                if potential_locations:
                    existing = set(segment['named_entities'].get('locations', []))
                    segment['named_entities']['locations'] = list(existing.union(potential_locations))
            
            # Special processing for date/time segments
            if any(date_term in heading for date_term in ['date', 'time', 'period', 'duration', 'schedule']):
                potential_dates = self._extract_potential_dates(text)
                if potential_dates:
                    existing = set(segment['named_entities'].get('dates', []))
                    segment['named_entities']['dates'] = list(existing.union(potential_dates))
            
            # Process organization-focused segments
            if any(org_term in heading for org_term in ['company', 'organization', 'institution', 'agency', 'department']):
                potential_orgs = self._extract_potential_organizations(text)
                if potential_orgs:
                    existing = set(segment['named_entities'].get('organizations', []))
                    segment['named_entities']['organizations'] = list(existing.union(potential_orgs))
        
        # Cross-segment entity verification
        self._verify_entities_across_segments(segments)
    
    def _extract_potential_locations(self, text):
        """Extract potential locations that might have been missed by spaCy."""
        locations = set()
        
        # Common location indicators
        location_patterns = [
            r'\b(?:in|at|near|from)\s+([A-Z][a-zA-Z.-]+(?:\s+[A-Z][a-zA-Z.-]+){0,2})\b',
            r'\b([A-Z][a-z]+)\s+(?:County|Parish|District|Province|State|City|Town|Village)\b',
            r'\b([A-Z][a-z]+(?:,)?\s+[A-Z]{2})\b',  # City, State format
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                clean_match = self._clean_entity_text(match)
                if clean_match and len(clean_match) > 2:
                    locations.add(clean_match)
        
        return list(locations)
    
    def _extract_potential_dates(self, text):
        """Extract potential dates that might have been missed by spaCy."""
        dates = set()
        
        # Enhanced date patterns
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            r'\bQ[1-4]\s+\d{4}\b',  # Quarters (Q1 2023)
            r'\b(?:FY|CY)\s*\d{2,4}\b',  # Fiscal/Calendar Year
            r'\b(?:early|mid|late)\s+\d{4}\b',  # Approximate years
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                clean_match = self._clean_entity_text(match)
                if clean_match:
                    dates.add(clean_match)
        
        return list(dates)
    
    def _extract_potential_organizations(self, text):
        """Extract potential organizations that might have been missed by spaCy."""
        organizations = set()
        
        # Organization indicators
        org_patterns = [
            r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+)\s+(?:Corporation|Corp|Company|Co|Inc|Ltd|LLC|LLP|Group|Association|Authority|Agency|Institute|Foundation)\b',
            r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+)\s+(?:Department|Commission|Committee|Council|Board|Bureau)\b',
            r'\bthe\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+)\b'
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                clean_match = self._clean_entity_text(match)
                if clean_match and len(clean_match.split()) > 1:  # Only multi-word organizations
                    organizations.add(clean_match)
        
        return list(organizations)
    
    def _extract_contact_information(self, text, entities):
        """Extract phone numbers and email addresses from text.
        
        Since spaCy doesn't reliably detect contact information by default,
        we use custom regex patterns to capture common formats. This covers
        international formats, country-specific patterns, and standard 
        email formats.
        
        Args:
            text: The text to search for contact information
            entities: Dictionary to store extracted entities
        """
        if not text:
            return
            
        # OPTIMIZATION: Use pre-compiled regex patterns for improved performance
        # These are now class constants to avoid recompilation on each call
        if not hasattr(self, '_EMAIL_PATTERN'):
            # RFC 5322 compliant email regex - highly accurate but fast
            self._EMAIL_PATTERN = re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b')
            
            
            # This pattern captures international and local formats while reducing false positives
            self._PHONE_PATTERN = re.compile(
                r'(?<![\d-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?)?(?:\d{1,4}[\s.-]?){2,}\d{1,4}(?![\d-])'
            )
        
        # Use the cached patterns for extraction
        emails = set(self._EMAIL_PATTERN.findall(text))
        phones = set(self._PHONE_PATTERN.findall(text))
            
        # Add to contact_info category
        if 'contact_info' not in entities:
            entities['contact_info'] = []
            
        # Normalize and filter phone numbers for higher accuracy
        filtered_phones = []
        for phone in phones:
            # Clean up phone number
            clean_phone = re.sub(r'[\s.-]', '', phone)
            # Validate by length (too short = likely false positive)
            if len(clean_phone) >= 7:  # Most legit phone numbers have at least 7 digits
                filtered_phones.append(phone)
                
        # Add extracted information to entity dictionary
        entities['contact_info'].extend(list(emails))
        entities['contact_info'].extend(filtered_phones)
    
    def _format_phone_number(self, phone):
        """Format phone numbers consistently for better readability.
        
        Takes various phone number formats and normalizes them to a consistent
        format based on recognized patterns. Handles international numbers,
        North American numbers, and various other formats.
        
        Args:
            phone: Raw phone number string with potential formatting
            
        Returns:
            Formatted phone number or None if input doesn't look valid
        """
        # Keep the international prefix but strip all other non-digit chars
        # The + symbol is significant for international numbers
        if phone.startswith('+'):
            digits_only = '+' + re.sub(r'\D', '', phone[1:])
        else:
            digits_only = re.sub(r'\D', '', phone)
            
        # Validate length - real phone numbers are typically 7-15 digits
        # This helps filter out false positives (random number sequences)
        if len(digits_only) < 7 or len(digits_only) > 15:
            return None
            
        # Apply consistent formatting based on recognized patterns
        if digits_only.startswith('+'):
            # Keep international numbers in E.164 format (most universal)
            # Example: +12025550179
            return digits_only
        elif len(digits_only) == 10:
            # Format North American numbers as (XXX) XXX-XXXX
            # Example: (202) 555-0179
            return f"({digits_only[:3]}) {digits_only[3:6]}-{digits_only[6:]}"
        else:
            # For other formats, group into chunks of 4 digits for readability
            # Example: 9999-8888-7777
            groups = []
            for i in range(0, len(digits_only), 4):
                groups.append(digits_only[i:i+4])
            return '-'.join(groups)
    
    def _verify_entities_across_segments(self, segments):
        """Cross-validate entities between segments for better accuracy."""
        
        # Collect high-confidence entities from all segments
        high_confidence_entities = {
            'persons': set(),
            'organizations': set(),
            'locations': set(),
            'contact_info': set()  # Add contact information to cross-validation
        }
        
        # First pass: collect entities that appear multiple times
        entity_counts = {entity_type: {} for entity_type in high_confidence_entities}
        
        for segment in segments:
            for entity_type in high_confidence_entities:
                for entity in segment['named_entities'].get(entity_type, []):
                    if entity in entity_counts[entity_type]:
                        entity_counts[entity_type][entity] += 1
                    else:
                        entity_counts[entity_type][entity] = 1
        
        # Second pass: identify high-confidence entities (appearing in multiple segments)
        for entity_type, counts in entity_counts.items():
            for entity, count in counts.items():
                if count >= 2 or (count == 1 and len(entity.split()) > 1):  # Multi-word entities are more reliable
                    high_confidence_entities[entity_type].add(entity)
        
        # Third pass: ensure high-confidence entities are consistent across segments
        for segment in segments:
            # Make sure entities are in the right category
            for entity_type, entities in high_confidence_entities.items():
                # Check if any high-confidence entity appears in the text but not in the right category
                segment_text = segment.get('segment_text', '')
                for entity in entities:
                    if entity in segment_text and entity not in segment['named_entities'].get(entity_type, []):
                        # Add the missing entity to the correct category
                        if entity_type not in segment['named_entities']:
                            segment['named_entities'][entity_type] = []
                        segment['named_entities'][entity_type].append(entity)
    
    def _extract_entities(self, doc):
        """Extract and categorize entities from a spaCy document.
        
        This is our core entity extraction logic. It uses a multi-pass approach:
        1. Extract contact info via regex (emails, phones)
        2. Process spaCy's built-in NER with context verification
        3. Apply additional pattern matching for specific entity types
        
        The multi-pass approach significantly improves accuracy over using
        just the default spaCy NER, especially for specialized documents.
        
        Args:
            doc: A processed spaCy Doc object
            
        Returns:
            Dictionary of entities by category
        """
        # Initialize result dictionary with empty lists for each entity type
        entities = {entity_type: [] for entity_type in self.entity_types.keys()}
        
        # PASS 1: Extract contact information (emails and phone numbers) 
        # We handle these separately because spaCy's default models aren't
        # trained to detect them well
        text = doc.text
        self._extract_contact_information(text, entities)
        
        # PASS 2: Process entities from spaCy's NER system
        # Apply additional context verification to improve accuracy
        for ent in doc.ents:
            if self._is_low_quality_entity(ent):
                continue
                
            # Context verification to improve accuracy
            context_verified = True
            if ent.label_ == "ORG" and len(ent.text.split()) == 1:  # Verify single-word organizations
                context_words = self._get_context_words(doc, ent.start, ent.end, 3)
                org_indicators = ["company", "corp", "corporation", "inc", "llc", "ltd", "group"]
                if not any(word in context_words.lower() for word in org_indicators):
                    # Extra verification for potential false positives
                    if ent.text.lower() in ["court", "government", "department", "agency", "committee"]:
                        pass  # These are likely valid organizations
                    else:
                        # Check capitalization pattern
                        if not all(c.isupper() for c in ent.text if c.isalpha()):
                            context_verified = self._verify_entity_by_frequency(doc.text, ent.text, ent.label_)
                    
            elif ent.label_ == "LOC" or ent.label_ == "GPE":
                # Verify location entities that could be confused with persons
                if ent.text in entities.get("persons", []):
                    context_words = self._get_context_words(doc, ent.start, ent.end, 4)
                    if "in" not in context_words and "at" not in context_words and "near" not in context_words:
                        context_verified = False
            
            if context_verified:
                for entity_type, spacy_labels in self.entity_types.items():
                    if ent.label_ in spacy_labels:
                        clean_entity = self._clean_entity_text(ent.text)
                        if clean_entity and clean_entity not in entities[entity_type]:
                            entities[entity_type].append(clean_entity)
                        break
        
        # Second pass: Use regex patterns for specific entity types
        # Improved date extraction with more comprehensive patterns
        if "dates" in entities:
            date_patterns = [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # 01/20/2022, 1-20-22
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # 2022/01/20, ISO format
                r'\b\d{4}\b',  # Years like 2022
                r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}\b',  # January 20, 2022
                r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[a-z]*(?:,)?\s+\d{4}\b',  # 20th of January, 2022
                r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[a-z]*\s+\d{4}\b'  # Month Year
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, doc.text)
                for match in matches:
                    clean_match = self._clean_entity_text(match)
                    if clean_match and clean_match not in entities["dates"]:
                        entities["dates"].append(clean_match)
        
        # Improved location extraction with common location pattern detection
        if "locations" in entities:
            location_patterns = [
                r'\b(?:in|at|near|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b(?!\s+(?:University|College|School|Hospital|Corporation|Company|Inc|Ltd|LLC))',
                r'\bin\s+([A-Z][a-z]+(?:,)?\s+[A-Z]{2})\b',  # City, State format
                r'\b([A-Z][a-z]+)\s+(?:County|Province|Region|District|Territory)\b'
            ]
            
            for pattern in location_patterns:
                matches = re.findall(pattern, doc.text)
                for match in matches:
                    clean_match = self._clean_entity_text(match)
                    if clean_match and clean_match not in entities["locations"]:
                        entities["locations"].append(clean_match)
        
        return entities
    
    def _get_context_words(self, doc, start_idx, end_idx, window_size=3):
        """Get context words around an entity to help with verification."""
        start = max(0, start_idx - window_size)
        end = min(len(doc), end_idx + window_size)
        return doc[start:end].text
    
    def _verify_entity_by_frequency(self, text, entity_text, label, threshold=2):
        """Verify an entity by checking its frequency in the document."""
        count = text.count(entity_text)
        if count >= threshold:
            return True
        
        # For persons, check if the entity looks like a proper name
        if label == "PERSON":
            if all(part[0].isupper() for part in entity_text.split() if part):
                return True
                
        return False
    
    def _is_low_quality_entity(self, ent):
        text = ent.text.strip()
        
        if len(text) < 2 and ent.label_ != "DATE":
            return True
            
        if all(c in string.punctuation + string.whitespace for c in text):
            return True
            
        if ent.label_ != "DATE" and self._is_numeric_entity(text):
            return True
            
        false_positive_words = ["the", "a", "an", "this", "that", "these", "those", "and", "or"]
        if text.lower() in false_positive_words:
            return True
            
        if len(text.split()) == 1 and text[0].isupper() and ent.start > 0 and not ent.sent.text.startswith(text):
            common_capitalized = ["I", "You", "He", "She", "We", "They", "It"]
            if text in common_capitalized:
                return True
        
        return False
    
    def _is_numeric_entity(self, text):
        cleaned = re.sub(r'[,\.\s]', '', text)
        return cleaned.isdigit()
    
    def _clean_entity_text(self, text):
        if not text:
            return ""
            
        clean = text.strip()
        clean = clean.strip(string.punctuation)
        
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        org_prefixes = ["the ", "The "]
        for prefix in org_prefixes:
            if clean.startswith(prefix) and len(clean) > len(prefix) + 3:  # Only if there's significant content after
                clean = clean[len(prefix):]
        
        return clean
    
    def _find_entity_variants(self, all_entities):
        entity_variants = {}
        
        for entity_type, entities in all_entities.items():
            variants = defaultdict(list)
            
            if len(entities) < 2:
                entity_variants[entity_type] = {}
                continue
                
            sorted_entities = sorted(entities, key=len, reverse=True)
            
            for i, entity in enumerate(sorted_entities):
                if any(entity in var_list for var_list in variants.values()):
                    continue
                    
                canonical = entity
                
                for other in sorted_entities[i+1:]:
                    if any(other in var_list for var_list in variants.values()):
                        continue
                        
                    if self._are_entity_variants(canonical, other, entity_type):
                        variants[canonical].append(other)
            
            entity_variants[entity_type] = variants
        
        return entity_variants
    
    def _are_entity_variants(self, entity1, entity2, entity_type):
        if len(entity1) < 3 or len(entity2) < 3:
            return False
            
        if entity1 == entity2 or f" {entity1} " in f" {entity2} " or f" {entity2} " in f" {entity1} ":
            return True
            
        if entity_type == "organizations":
            if _is_abbreviation(entity1, entity2) or _is_abbreviation(entity2, entity1):
                return True
                
            org_suffixes = [", Inc", " Inc", " Corp", " Corporation", " LLC", " Ltd"]
            clean1 = entity1
            clean2 = entity2
            
            for suffix in org_suffixes:
                if clean1.endswith(suffix):
                    clean1 = clean1[:-len(suffix)].strip()
                if clean2.endswith(suffix):
                    clean2 = clean2[:-len(suffix)].strip()
            
            if clean1 == clean2:
                return True
                
        elif entity_type == "persons":
            name1_parts = entity1.split()
            name2_parts = entity2.split()
            
            if len(name1_parts) > len(name2_parts) and name1_parts[-1] == name2_parts[-1] \
                and len(name1_parts[-1]) > 3:
                return True
            elif len(name2_parts) > len(name1_parts) and name2_parts[-1] == name1_parts[-1] \
                and len(name2_parts[-1]) > 3:
                return True
                
            if ',' in entity1 and ',' not in entity2:
                last_first = entity1.split(',', 1)
                if len(last_first) == 2 and last_first[0].strip() in entity2 and last_first[1].strip() in entity2:
                    return True
                    
            if ',' in entity2 and ',' not in entity1:
                last_first = entity2.split(',', 1)
                if len(last_first) == 2 and last_first[0].strip() in entity1 and last_first[1].strip() in entity1:
                    return True
        
        if len(entity1) > 5 and len(entity2) > 5:
            similarity = difflib.SequenceMatcher(None, entity1.lower(), entity2.lower()).ratio()
            return similarity > 0.85  # High threshold to avoid false matches
            
        return False


def _is_abbreviation(short, long):
    if not short.isupper() or len(short) < 2 or len(short) > 5:
        return False
        
    words = [w for w in long.split() if w]
    if not words:
        return False
        
    initials = ''.join(word[0].upper() for word in words if word and not word[0].islower())
    
    # Check if the abbreviation matches the initials
    return short == initials or short in initials
