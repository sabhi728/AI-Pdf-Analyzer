# 📄 Advanced Document Processor

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.21.0-FF4B4B)](https://streamlit.io/)
[![spaCy](https://img.shields.io/badge/spaCy-3.5.0-09A3D5)](https://spacy.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An intelligent document processing system that extracts structured information from PDFs with advanced segmentation and named entity recognition.

## 🚀 Features

<details open>
<summary><strong>Core Capabilities</strong></summary>

- **PDF Text Extraction** - Extract text while preserving document structure
- **Automatic Layout Analysis** - Detect columns, headers, footers, and tables
- **Hierarchical Segmentation** - Organize text into logical hierarchical sections 
- **Named Entity Recognition** - Identify and classify entities across segments
- **Metadata Extraction** - Extract dates, sources, and page mappings
- **Interactive Dashboard** - Visualize and explore extracted information
</details>

<details>
<summary><strong>Advanced Features</strong></summary>

- **Smart Header/Footer Removal** - Automatically identify and remove repeating elements
- **Multi-Column Detection** - Process documents with complex layouts
- **Entity Normalization** - Resolve entity variations and co-references
- **Natural Paragraph Boundaries** - Ensure text isn't cut off mid-paragraph
- **Optimized Performance** - Process large documents efficiently
- **Cached Processing** - Reuse results for faster repeated analysis
</details>

## 💻 Technologies

| Component | Technology | Description |
|-----------|------------|-------------|
| **Backend** | Python 3.8+ | Core processing logic and algorithms |
| **PDF Processing** | pdfplumber | Extract text with layout awareness |
| **NLP** | spaCy | Named entity recognition and linguistic analysis |
| **Text Processing** | NLTK | Natural language toolkit for text manipulation |
| **Web Interface** | Streamlit | Interactive web dashboard |
| **Data Structure** | JSON | Structured data output format |

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/sabhi728/AI-Pdf-Analyzer

# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg
```

## 📋 Usage

### Command Line Interface

```bash
python main.py --input path/to/your/document.pdf --output results.json
```

### Options

- `--input`, `-i`: Path to the input PDF file (required)
- `--output`, `-o`: Path to save the output file (required)
- `--format`, `-f`: Output format, either 'json' or 'csv' (default: 'json')
- `--spacy-model`: SpaCy model to use for NER (default: 'en_core_web_lg')

### Web Interface

```bash
python -m streamlit run app.py
```

The application will be available at `http://localhost:8501`.

### Processing a Document

<details open>
<summary><strong>Step-by-Step Guide</strong></summary>

1. **Upload a PDF Document**
   - Click the "Browse files" button to upload your PDF

2. **Configure Processing Options**
   - The system uses optimized default settings for most documents
   - Advanced options are available in the command line interface

3. **Process the Document**
   - Click "Process Document" to start analysis
   - The system will extract text, analyze structure, and identify entities in three steps

4. **Explore Results**
   - Navigate between Document Overview and Raw Data tabs
   - Download the structured JSON output for further use
</details>

## 🏗️ Architecture

```
PDF Document → Document Reader → Document Segmenter → NER Processor → JSON Output → Dashboard
     ↓               ↓                 ↓                  ↓               ↓           ↓
   Input      Text Extraction    Segment Hierarchy    Entity Detection   Output     Visualization
```

### Core Components

<details open>
<summary><strong>Document Reader</strong> (document_reader.py)</summary>

Handles PDF parsing and text extraction with layout awareness:
- Detects and removes headers/footers
- Identifies complex layouts including multi-column format
- Preserves document structure and page mapping
</details>

<details>
<summary><strong>Document Segmenter</strong> (segmentation.py)</summary>

Organizes text into logical segments:
- Identifies headings using pattern matching and heuristics
- Builds hierarchical document structure
- Maintains proper segment boundaries
- Extracts metadata such as dates and sources
</details>

<details>
<summary><strong>NER Processor</strong> (ner_processor.py)</summary>

Performs entity extraction and analysis:
- Uses spaCy for named entity recognition
- Categorizes entities into types (persons, organizations, etc.)
- Normalizes entity variations
- Optimizes processing using batching and caching
</details>

<details>
<summary><strong>Web Dashboard</strong> (app.py)</summary>

Provides an interactive interface built with Streamlit:
- File upload and processing controls
- Document overview visualization
- Raw data exploration
- JSON data export
</details>

## 📊 Example Output

```json
{
  "segments": [
    {
      "segment_level": 1,
      "segment_title": "Executive Summary",
      "segment_text": "This report provides an analysis of...",
      "segment_date": "2023-05-15",
      "segment_source": "Research Department",
      "start_index": 0,
      "end_index": 1250,
      "pages": [1, 2],
      "named_entities": {
        "persons": ["John Smith", "Sarah Johnson"],
        "organizations": ["Acme Corporation", "Global Institute"],
        "locations": ["New York", "London"],
        "dates": ["2023", "May 15, 2023"],
        "misc": ["Report A-123"]
      }
    }
  ]
}
```

## 🔍 Performance Optimizations

<details>
<summary><strong>Show optimization techniques</strong></summary>

- **PDF Extraction**
  - Simplified text block extraction for faster processing
  - Optimized header/footer detection
  - Smart column detection

- **Document Segmentation**
  - Fast heading pattern recognition
  - Efficient hierarchical structure building
  - Natural paragraph boundary detection

- **Entity Recognition**
  - Batch processing with small efficient batches
  - Entity caching for repeated segments
  - Simplified normalization for large documents
  - Text length limiting for consistent performance
</details>

## 🧠 Technical Considerations

1. The document has a logical structure with headings that can be identified by patterns
2. The text extraction quality depends on the PDF's structure and quality
3. The segmentation algorithm works best with documents that follow standard formatting conventions
4. The NER performance depends on the spaCy model used and the quality of the text

## ⚠️ Error Handling

The solution includes comprehensive error handling for:
- File not found errors
- PDF parsing errors
- Text extraction failures
- Processing errors during segmentation or NER



