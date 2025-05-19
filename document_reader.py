import pdfplumber
import os
import logging
import re
from collections import defaultdict

class DocumentReader:
    
    def __init__(self):
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
        if not os.path.exists(pdf_path):
            self.logger.error(f"File not found: {pdf_path}")
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")
        
        try:
            full_text = ""
            page_metadata = []
            
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
                
                
                for i, page in enumerate(pdf.pages):
                    
                    if self.layout_info['multi_column'] and detect_columns:
                        text_blocks = self._extract_text_blocks(page, simplified=True)
                        page_text = self._extract_text_from_columns(page, text_blocks)
                    else:
                        page_text = page.extract_text()
                    
                    if page_text:
                        if remove_headers_footers:
                            page_text = self._remove_headers_footers(page_text, i)
                        
                        start_index = len(full_text)
                        full_text += page_text + "\n\n"
                        end_index = len(full_text)
                        
                        page_metadata.append({
                            'page_number': i + 1,  
                            'start_index': start_index,
                            'end_index': end_index,
                            'text_length': len(page_text)
                        })
            
            self.layout_info['has_header'] = any(h['is_header'] for h in self.header_footer_candidates)
            self.layout_info['has_footer'] = any(h['is_footer'] for h in self.header_footer_candidates)
            
            self.logger.info(f"Extraction complete: {len(full_text)} characters from {len(page_metadata)} pages")
            return full_text.strip(), self.layout_info, page_metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
            raise
    
    def _extract_text_blocks(self, page, simplified=False):
        try:
            if simplified:
                text = page.extract_text()
                if not text:
                    return []
                return [{
                    "text": text,
                    "bbox": (0, 0, page.width, page.height)
                }]
            
            page_words = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
            
            if not page_words:
                return []
                
            lines = {}
            line_tolerance = 5  
            
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
