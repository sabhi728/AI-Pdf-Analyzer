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
    
    def __init__(self, model="en_core_web_lg", use_transformers=False):
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
        
        
        try:
            self.logger.info(f"Loading spaCy model: {model}")
            
            
            if use_transformers and not model.startswith("xx_"):  # xx_ prefix is for multi-lingual models
                try:
                    
                    import spacy_transformers
                    self.nlp = spacy.load(model)
                    self.logger.info("Using transformer-based NER pipeline")
                except (ImportError, OSError):
                    self.logger.warning("Transformer components not available. Using standard model.")
                    self.nlp = spacy.load(model)
            else:
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
        self.logger.info("Starting high-performance NER processing")
        
        if not segments:
            return []
            
        skip_normalization = len(segments) > 20
        
        for segment in segments:
            if 'named_entities' not in segment:
                segment['named_entities'] = {
                    "persons": [], "organizations": [], "locations": [],
                    "dates": [], "misc": []
                }
                
        batch_size = 5  # Process 5 segments at a time
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            self._process_segment_batch_fast(batch)
        
        if not skip_normalization:
            self._normalize_entities_fast(segments)
        
        return segments
    
    def _process_segment_batch_fast(self, segments):
        segments_to_process = []
        for i, segment in enumerate(segments):
            if len(segment.get('segment_text', '')) < 50:
                segment['named_entities'] = {
                    "persons": [], "organizations": [], "locations": [],
                    "dates": [], "misc": []
                }
                continue
                
            text_hash = hash(segment.get('segment_text', '')[:100])  # Use first 100 chars for hashing
            if text_hash in self.entity_cache:
                segment['named_entities'] = self.entity_cache[text_hash].copy()
            else:
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
                with self.nlp.select_pipes(disable=["lemmatizer", "attribute_ruler"]):
                    max_chars = min(len(text), 10000)  # Limit to first 10K chars for speed
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
        
        for entity_type in ['persons', 'organizations']:
            all_entities = set()
            for segment in segments:
                if entity_type in segment['named_entities']:
                    all_entities.update(segment['named_entities'][entity_type])
                
            if len(all_entities) < 5:
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
    
    def _extract_entities(self, doc):
        entities = {entity_type: [] for entity_type in self.entity_types.keys()}
        
        for ent in doc.ents:
            if self._is_low_quality_entity(ent):
                continue
                
            for entity_type, spacy_labels in self.entity_types.items():
                if ent.label_ in spacy_labels:
                    clean_entity = self._clean_entity_text(ent.text)
                    if clean_entity and clean_entity not in entities[entity_type]:
                        entities[entity_type].append(clean_entity)
                    break
        
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
