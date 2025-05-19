import spacy
import logging
from tqdm import tqdm
import re
import difflib
from collections import defaultdict, Counter
import string
from spacy.tokens import Doc, Span
from spacy.language import Language

class NERProcessor:
    """Enhanced class for named entity recognition with advanced techniques."""
    
    def __init__(self, model="en_core_web_lg", use_transformers=False):
        """
        Initialize the NER processor with advanced options.
        
        Args:
            model (str): Name of the spaCy model to use.
            use_transformers (bool): Whether to use transformer models for better accuracy (slower).
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.entity_confidence_threshold = 0.7
        self.context_window_size = 5
        self.use_transformers = use_transformers
        self.entity_cache = {}
        
        self.entity_types = {
            "persons": ["PERSON"],
            "organizations": ["ORG", "NORP", "FAC"],
            "locations": ["GPE", "LOC"],
            "dates": ["DATE"],
            "misc": ["MISC", "QUANTITY", "CARDINAL", "ORDINAL"],
            "monetary": ["MONEY"],
            "percentages": ["PERCENT"],
            "products": ["PRODUCT"],
            "events": ["EVENT"],
            "laws": ["LAW"],
            "works": ["WORK_OF_ART"]
        }
        
        # Try to load the specified model
        try:
            self.logger.info(f"Loading spaCy model: {model}")
            
            # If using transformers, use a pipeline with transformer components
            if use_transformers and not model.startswith("xx_"):  # xx_ prefix is for multi-lingual models
                try:
                    # Try to load a transformer-based pipeline
                    import spacy_transformers
                    self.nlp = spacy.load(model)
                    self.logger.info("Using transformer-based NER pipeline")
                except (ImportError, OSError):
                    self.logger.warning("Transformer components not available. Using standard model.")
                    self.nlp = spacy.load(model)
            else:
                self.nlp = spacy.load(model)
                
            self.logger.info("SpaCy model loaded successfully")
            
            # Register custom components
            self._register_custom_components()
            
        except OSError:
            self.logger.warning(f"SpaCy model {model} not found. Downloading...")
            spacy.cli.download(model)
            self.nlp = spacy.load(model)
            self._register_custom_components()
            self.logger.info(f"SpaCy model {model} downloaded and loaded")
    
    def _register_custom_components(self):
        """Register custom pipeline components for improved entity recognition."""
        # Add custom pipeline components if they don't already exist
        if "entity_cleaner" not in self.nlp.pipe_names:
            # Custom component to clean up entity spans
            @Language.component("entity_cleaner")
            def entity_cleaner(doc):
                # Improve entity recognition by cleaning up bad spans
                if not doc.ents:
                    return doc
                    
                new_ents = []
                for ent in doc.ents:
                    # Skip numeric entities unless they're dates
                    if ent.label_ != "DATE" and self._is_numeric_entity(ent.text):
                        continue
                        
                    # Clean up entity spans (remove punctuation at boundaries)
                    cleaned_start = ent.start
                    cleaned_end = ent.end
                    
                    # Remove leading/trailing punctuation from entity spans
                    span_text = ent.text
                    while cleaned_start < cleaned_end and span_text.startswith(tuple(string.punctuation)):
                        cleaned_start += 1
                        span_text = doc[cleaned_start:cleaned_end].text
                        
                    while cleaned_start < cleaned_end and span_text.endswith(tuple(string.punctuation)):
                        cleaned_end -= 1
                        span_text = doc[cleaned_start:cleaned_end].text
                    
                    # Only keep the entity if it still has content after cleaning
                    if cleaned_start < cleaned_end:
                        new_span = Span(doc, cleaned_start, cleaned_end, label=ent.label_)
                        new_ents.append(new_span)
                        
                # Set the cleaned entities
                doc.ents = new_ents
                return doc
                
            # Add the custom component to the pipeline
            self.nlp.add_pipe("entity_cleaner", last=True)
            self.logger.info("Added custom entity cleaning component to the pipeline")
    
    def process_segments(self, segments):
        """
        Process segments to extract named entities with ultra-fast performance.
        
        Args:
            segments (list): List of segment dictionaries.
            
        Returns:
            list: Segments with extracted named entities.
        """
        self.logger.info("Starting high-performance NER processing")
        
        # Early exit for empty input
        if not segments:
            return []
            
        # Performance optimization: Skip normalization for large documents
        skip_normalization = len(segments) > 20
        
        # Make sure all segments have named_entities field initialized
        for segment in segments:
            if 'named_entities' not in segment:
                segment['named_entities'] = {
                    "persons": [], "organizations": [], "locations": [],
                    "dates": [], "misc": []
                }
                
        # Fast path: Process segments in small batches for better performance
        # This reduces memory usage and speeds up processing
        batch_size = 5  # Process 5 segments at a time
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            self._process_segment_batch_fast(batch)
        
        # Skip normalization for large documents (big performance gain)
        if not skip_normalization:
            self._normalize_entities_fast(segments)
        
        return segments
    def _process_segment_batch_fast(self, segments):
        """Process a batch of segments with optimized performance."""
        # Skip empty segments
        segments_to_process = []
        for i, segment in enumerate(segments):
            # Skip segments that are too short (or initialize with empty entities)
            if len(segment.get('segment_text', '')) < 50:
                segment['named_entities'] = {
                    "persons": [], "organizations": [], "locations": [],
                    "dates": [], "misc": []
                }
                continue
                
            # Check cache first for very fast processing
            text_hash = hash(segment.get('segment_text', '')[:100])  # Use first 100 chars for hashing
            if text_hash in self.entity_cache:
                segment['named_entities'] = self.entity_cache[text_hash].copy()
            else:
                segments_to_process.append((i, segment, text_hash))
            
        # Process remaining segments
        if not segments_to_process:
            return
            
        # Process each segment individually for reliable results
        for i, segment, text_hash in segments_to_process:
            text = segment.get('segment_text', '')
            if not text:
                segment['named_entities'] = {
                    "persons": [], "organizations": [], "locations": [],
                    "dates": [], "misc": []
                }
                continue
                
            try:
                # Use a streamlined pipeline with fewer components for speed
                with self.nlp.select_pipes(disable=["lemmatizer", "attribute_ruler"]):
                    # Limit text length for faster processing
                    max_chars = min(len(text), 10000)  # Limit to first 10K chars for speed
                    doc = self.nlp(text[:max_chars])
                
                # Extract entities
                entities = self._extract_entities(doc)
                segment['named_entities'] = entities
                self.entity_cache[text_hash] = entities.copy()
            except Exception as e:
                self.logger.warning(f"Error processing segment: {str(e)}")
                # Ensure segment has empty entity structure
                segment['named_entities'] = {
                    "persons": [], "organizations": [], "locations": [],
                    "dates": [], "misc": []
                }
        
    def _normalize_entities_fast(self, segments):
        """Simplified entity normalization focused on speed."""
        self.logger.info("Fast entity normalization") 
        
        # Only normalize important entity types
        for entity_type in ['persons', 'organizations']:
            # Get all entities of this type
            all_entities = set()
            for segment in segments:
                if entity_type in segment['named_entities']:
                    all_entities.update(segment['named_entities'][entity_type])
                
            # Skip if very few entities
            if len(all_entities) < 5:
                continue
                
            # Simple normalization: map case variants
            entity_map = {}
            for entity in all_entities:
                lower = entity.lower()
                if lower not in entity_map:
                    entity_map[lower] = entity
                elif len(entity) > len(entity_map[lower]):
                    # Keep the longer variant
                    entity_map[lower] = entity
            
            # Apply normalization
            for segment in segments:
                if entity_type in segment['named_entities']:
                    normalized = []
                    for entity in segment['named_entities'][entity_type]:
                        normalized.append(entity_map.get(entity.lower(), entity))
                    segment['named_entities'][entity_type] = normalized
        
        self.logger.info("Enhanced named entity recognition complete")
    
    def _extract_entities(self, doc):
        """
        Extract and classify entities from a spaCy document.
        
        Args:
            doc: spaCy Doc object.
            
        Returns:
            dict: Dictionary of entity lists by type.
        """
        entities = {entity_type: [] for entity_type in self.entity_types.keys()}
        
        # Process standard named entities
        for ent in doc.ents:
            # Skip low-quality entities
            if self._is_low_quality_entity(ent):
                continue
                
            # Map to our custom entity types
            for entity_type, spacy_labels in self.entity_types.items():
                if ent.label_ in spacy_labels:
                    clean_entity = self._clean_entity_text(ent.text)
                    if clean_entity and clean_entity not in entities[entity_type]:
                        entities[entity_type].append(clean_entity)
                    break
        
        # Add special handling for dates that might be missed by spaCy
        if "dates" in entities:
            date_patterns = [
                r'\b\d{4}\b',  # Years like 2022
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b'  # Month Year
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, doc.text)
                for match in matches:
                    clean_match = self._clean_entity_text(match)
                    if clean_match and clean_match not in entities["dates"]:
                        entities["dates"].append(clean_match)
        
        return entities
    
    def _is_low_quality_entity(self, ent):
        """
        Check if an entity is likely to be a false positive or low quality.
        
        Args:
            ent: spaCy entity Span.
            
        Returns:
            bool: True if the entity should be filtered out.
        """
        text = ent.text.strip()
        
        # Skip very short entities except for dates
        if len(text) < 2 and ent.label_ != "DATE":
            return True
            
        # Skip entities that are just punctuation or whitespace
        if all(c in string.punctuation + string.whitespace for c in text):
            return True
            
        # Skip entities that are just numbers (except dates)
        if ent.label_ != "DATE" and self._is_numeric_entity(text):
            return True
            
        # Skip common false positives
        false_positive_words = ["the", "a", "an", "this", "that", "these", "those", "and", "or"]
        if text.lower() in false_positive_words:
            return True
            
        # Check if the entity is just an isolated capitalized word in a sentence
        if len(text.split()) == 1 and text[0].isupper() and ent.start > 0 and not ent.sent.text.startswith(text):
            # Check if this is a pronoun or common word
            common_capitalized = ["I", "You", "He", "She", "We", "They", "It"]
            if text in common_capitalized:
                return True
        
        return False
    
    def _is_numeric_entity(self, text):
        """
        Check if an entity is just a number.
        
        Args:
            text (str): Entity text.
            
        Returns:
            bool: True if the entity is just numeric.
        """
        # Remove punctuation and check if it's just digits
        cleaned = re.sub(r'[,\.\s]', '', text)
        return cleaned.isdigit()
    
    def _clean_entity_text(self, text):
        """
        Clean entity text by removing unwanted characters and normalization.
        
        Args:
            text (str): Entity text to clean.
            
        Returns:
            str: Cleaned entity text.
        """
        if not text:
            return ""
            
        # Remove leading/trailing whitespace and punctuation
        clean = text.strip()
        clean = clean.strip(string.punctuation)
        
        # Normalize whitespace
        clean = re.sub(r'\s+', ' ', clean).strip()
        
        # Specific cleaning for organization names
        # Remove common prefixes that aren't part of the actual name
        org_prefixes = ["the ", "The "]
        for prefix in org_prefixes:
            if clean.startswith(prefix) and len(clean) > len(prefix) + 3:  # Only if there's significant content after
                clean = clean[len(prefix):]
        
        return clean
    
    def _find_entity_variants(self, all_entities):
        """
        Find variations of the same entity across segments for normalization.
        
        Args:
            all_entities (dict): Dictionary of all entities by type.
            
        Returns:
            dict: Dictionary of entity variants by type.
        """
        entity_variants = {}
        
        for entity_type, entities in all_entities.items():
            variants = defaultdict(list)
            
            # Skip if too few entities
            if len(entities) < 2:
                entity_variants[entity_type] = {}
                continue
                
            # Sort entities by length (descending) to prefer longer forms as canonical
            sorted_entities = sorted(entities, key=len, reverse=True)
            
            for i, entity in enumerate(sorted_entities):
                # Skip if this entity is already a variant of another
                if any(entity in var_list for var_list in variants.values()):
                    continue
                    
                # This will be our canonical form
                canonical = entity
                
                # Look for variants
                for other in sorted_entities[i+1:]:
                    # Skip if other is already a variant
                    if any(other in var_list for var_list in variants.values()):
                        continue
                        
                    # Check if they're variants of each other
                    if self._are_entity_variants(canonical, other, entity_type):
                        variants[canonical].append(other)
            
            entity_variants[entity_type] = variants
        
        return entity_variants
    
    def _are_entity_variants(self, entity1, entity2, entity_type):
        """
        Check if two entities are variants of the same real-world entity.
        
        Args:
            entity1 (str): First entity text.
            entity2 (str): Second entity text.
            entity_type (str): Type of entity (persons, organizations, etc.)
            
        Returns:
            bool: True if they are likely variants of the same entity.
        """
        # Don't match very short strings
        if len(entity1) < 3 or len(entity2) < 3:
            return False
            
        # Exact match or one is contained within the other with word boundaries
        if entity1 == entity2 or f" {entity1} " in f" {entity2} " or f" {entity2} " in f" {entity1} ":
            return True
            
        # Type-specific variant rules
        if entity_type == "organizations":
            # Organization abbreviations (e.g., "Federal Bureau of Investigation" -> "FBI")
            if _is_abbreviation(entity1, entity2) or _is_abbreviation(entity2, entity1):
                return True
                
            # Handle organization suffixes (Corp, Inc, etc.)
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
            # Person name variants
            # E.g., "John Smith" vs "Dr. John Smith" or "Smith, John"
            
            # Try matching last names for potential matches
            name1_parts = entity1.split()
            name2_parts = entity2.split()
            
            # If one has more parts than the other, it might be a full name vs. just last name
            if len(name1_parts) > len(name2_parts) and name1_parts[-1] == name2_parts[-1] \
                and len(name1_parts[-1]) > 3:
                return True
            elif len(name2_parts) > len(name1_parts) and name2_parts[-1] == name1_parts[-1] \
                and len(name2_parts[-1]) > 3:
                return True
                
            # Check for "Lastname, Firstname" vs "Firstname Lastname"
            if ',' in entity1 and ',' not in entity2:
                last_first = entity1.split(',', 1)
                if len(last_first) == 2 and last_first[0].strip() in entity2 and last_first[1].strip() in entity2:
                    return True
                    
            if ',' in entity2 and ',' not in entity1:
                last_first = entity2.split(',', 1)
                if len(last_first) == 2 and last_first[0].strip() in entity1 and last_first[1].strip() in entity1:
                    return True
        
        # For other entity types, use string similarity
        if len(entity1) > 5 and len(entity2) > 5:
            # Calculate similarity ratio
            similarity = difflib.SequenceMatcher(None, entity1.lower(), entity2.lower()).ratio()
            return similarity > 0.85  # High threshold to avoid false matches
            
        return False


def _is_abbreviation(short, long):
    """
    Check if the short string is an abbreviation of the long string.
    
    Args:
        short (str): Potential abbreviation.
        long (str): Longer string.
        
    Returns:
        bool: True if short is an abbreviation of long.
    """
    # Only consider short strings with uppercase letters as potential abbreviations
    if not short.isupper() or len(short) < 2 or len(short) > 5:
        return False
        
    # Get the first letter of each word in the long string
    words = [w for w in long.split() if w]
    if not words:
        return False
        
    initials = ''.join(word[0].upper() for word in words if word and not word[0].islower())
    
    # Check if the abbreviation matches the initials
    return short == initials or short in initials
