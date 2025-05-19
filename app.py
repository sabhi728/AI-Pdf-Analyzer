import streamlit as st
import json
import os
import re
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from collections import Counter
from streamlit_option_menu import option_menu

from document_reader import DocumentReader
from segmentation import DocumentSegmenter
from ner_processor import NERProcessor


st.set_page_config(
    page_title="Advanced Document Analyzer",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    
    .hierarchy-level-1 {
        border-left: 4px solid #FF4B4B;
        padding-left: 10px;
    }
    
    .hierarchy-level-2 {
        border-left: 4px solid #0068C9;
        padding-left: 20px;
    }
    
    .hierarchy-level-3 {
        border-left: 4px solid #83C9FF;
        padding-left: 30px;
    }
    
    .hierarchy-level-4 {
        border-left: 4px solid #29B09D;
        padding-left: 40px;
    }
    
    .hierarchy-level-5 {
        border-left: 4px solid #FFBD45;
        padding-left: 50px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Top navigation menu
    selected = option_menu(
        menu_title=None,
        options=["Analyzer", "Documentation"],
        icons=["graph-up", "file-text"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "margin-bottom": "1rem"},
            "icon": {"font-size": "1rem"},
            "nav-link": {"font-size": "1rem", "text-align": "center", "padding": "0.5rem 1rem"}
        }
    )
    
    if selected == "Analyzer":
        show_analyzer()
    else:
        show_documentation()

def show_analyzer():
    st.title("Advanced Document Analyzer") 
    st.markdown("""
    This application uses AI to intelligently process PDF documents. It:
    
    1. **Extracts text** with advanced layout recognition (multi-column, headers/footers)
    2. **Identifies document structure** with hierarchical segmentation
    3. **Extracts and disambiguates entities** using NER techniques
    4. **Produces structured JSON data** based on the actual document content
    """)
    
    # Sidebar for document upload and processing controls
    with st.sidebar:
        st.header("📤 Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        # Output format info
        st.header("⚙️ Output Format")
        output_format = "JSON"  # Set JSON as the only format
        st.info("Data will be provided in structured JSON format")
        
        # Set sensible defaults for all parameters
        # These values are optimized for speed and accuracy
        spacy_model = "en_core_web_lg"
        remove_headers = True
        detect_columns = True
        min_confidence = 0.5
        use_transformers = False
        entity_threshold = 0.6
        
        # Process button
        process_col1, process_col2 = st.columns(2)
        with process_col1:
            process_button = st.button("🔍 Process Document", type="primary", use_container_width=True)
        with process_col2:
            reset_button = st.button("🔄 Reset", type="secondary", use_container_width=True)
        
        # App information
        st.header("ℹ️ About")
        st.info("""
        This advanced document analyzer uses AI to extract structured information from PDFs.
        
        Features:
        - Dynamic document structure identification
        - Advanced entity recognition and linking
        - Enhanced metadata extraction
        - Intelligent handling of complex layouts
        
        The system adapts to each document's unique structure.
        """)
    
    # Initialize session state for storing results
    if 'processed_segments' not in st.session_state:
        st.session_state.processed_segments = None
    if 'doc_layout_info' not in st.session_state:
        st.session_state.doc_layout_info = None
    if 'page_metadata' not in st.session_state:
        st.session_state.page_metadata = None
    if 'entity_stats' not in st.session_state:
        st.session_state.entity_stats = None
        
    # Reset function
    if reset_button:
        st.session_state.processed_segments = None
        st.session_state.doc_layout_info = None
        st.session_state.page_metadata = None
        st.session_state.entity_stats = None
        st.rerun()
    
    # Main content area for document processing and visualization
    if uploaded_file is not None:
        # Create tabs for different views of the processed document
        tabs = st.tabs(["💬 Document Overview", "📋 Raw Data"])
        
        # Create a temporary file to save the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name
        
        # Process the document when button is clicked
        if process_button or st.session_state.processed_segments is not None:
            # If we need to process the document
            if process_button or st.session_state.processed_segments is None:
                # Define status area outside the try block so it's always accessible
                status_area = st.empty()
                
                try:
                    with st.spinner("Step 1/3: Extracting text from PDF..."):
                        status_area.info("Step 1/3: Extracting text from PDF...")
                        reader = DocumentReader()
                        
                        @st.cache_data(show_spinner=False)
                        def extract_text(path, remove_headers, detect_cols):
                            try:
                                return reader.extract_text_from_pdf(path, remove_headers_footers=remove_headers, detect_columns=detect_cols)
                            except Exception as e:
                                print(f"PDF extraction error: {str(e)}")
                                raise Exception(f"Error extracting text: {str(e)}")
                        
                        text, layout_info, page_metadata = extract_text(pdf_path, remove_headers, detect_columns)
                        status_area.success("✅ Text extraction complete!")
                    
                    # Check if we have valid text to process
                    if not text or len(text.strip()) < 10:
                        raise Exception("The PDF appears to be empty or contains no extractable text")
                    
                    with st.spinner("Step 2/3: Analyzing document structure..."):
                        # Step 2: Segment the document with optimized detection
                        status_area.info("Step 2/3: Analyzing document structure...")
                        segmenter = DocumentSegmenter(use_machine_learning=False)  # Disable ML for speed
                        segmenter.heading_config['min_confidence'] = min_confidence
                        
                        @st.cache_data(show_spinner=False)
                        def segment_doc(txt, layout, metadata):
                            try:
                                return segmenter.segment_document(txt, layout, metadata)
                            except Exception as e:
                                print(f"Segmentation error: {str(e)}")
                                raise Exception(f"Error analyzing document structure: {str(e)}")
                        
                        segments = segment_doc(text, layout_info, page_metadata)
                        status_area.success("✅ Structure analysis complete!")
                    
                    with st.spinner("Step 3/3: Identifying named entities..."):
                        # Step 3: Process named entities with optimized recognition
                        status_area.info("Step 3/3: Identifying named entities...")
                        ner = NERProcessor(model=spacy_model, use_transformers=False)  # Disable transformers for speed
                        ner.entity_confidence_threshold = entity_threshold
                        
                        # Validate segments structure before processing
                        if not segments or not isinstance(segments, list):
                            raise Exception("Invalid document structure detected")
                        
                        # Initialize entity fields if missing
                        for segment in segments:
                            if 'named_entities' not in segment:
                                segment['named_entities'] = {
                                    "persons": [], "organizations": [], "locations": [], 
                                    "dates": [], "misc": []
                                }
                        
                        @st.cache_data(show_spinner=False)
                        def process_ner(segs):
                            try:
                                return ner.process_segments(segs)
                            except Exception as e:
                                print(f"NER processing error: {str(e)}")
                                raise Exception(f"Error extracting entities: {str(e)}")
                        
                        processed_segments = process_ner(segments)
                        
                        # Ensure all segments have proper entity fields
                        for segment in processed_segments:
                            for entity_type in ['persons', 'organizations', 'locations', 'dates', 'misc']:
                                if entity_type not in segment['named_entities']:
                                    segment['named_entities'][entity_type] = []
                        
                        status_area.success("✅ Entity extraction complete!")                        
                    
                    # Store results in session state
                    st.session_state.processed_segments = processed_segments
                    st.session_state.doc_layout_info = layout_info
                    st.session_state.page_metadata = page_metadata
                    
                    # Compute entity statistics for visualization (with caching)
                    @st.cache_data(show_spinner=False)
                    def compute_stats(segs):
                        try:
                            return compute_entity_statistics(segs)
                        except Exception as e:
                            print(f"Statistics computation error: {str(e)}")
                            # Return empty statistics as fallback
                            return {"total": 0, "by_type": {}, "by_segment": {}}
                        
                    entity_stats = compute_stats(processed_segments)
                    st.session_state.entity_stats = entity_stats
                    
                    # Show comprehensive success message
                    num_entities = sum(len(entity_list) for segment in processed_segments 
                                      for entity_type, entity_list in segment['named_entities'].items())
                    status_area.success(f"✨ Success! Processed document with {len(processed_segments)} segments and {num_entities} entities")
                
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    
                    # Log the detailed error for debugging
                    print(f"Error processing document: {error_details}")
                    
                    # Show user-friendly error message
                    status_area.error(f"❌ Error processing document: {str(e)}")
                    st.error("The document couldn't be processed. Please try a different PDF file.")
                    
                    # Provide more specific guidance based on error type
                    if "PDF syntax error" in str(e) or "file has not been decrypted" in str(e):
                        st.info("This PDF appears to be damaged, encrypted, or password-protected. Please try a different file.")
                    elif "Memory" in str(e) or "resource" in str(e).lower():
                        st.info("This PDF is too large or complex. Try using a smaller document.")
                    elif "entity" in str(e).lower() or "misc" in str(e).lower():
                        st.info("There was an issue with entity extraction. Processing will continue with simplified entity recognition.")
                    
                    # Clean up the temporary file
                    try:
                        os.unlink(pdf_path)
                    except:
                        pass
                    return
            
            # Display the processed results in different tabs
            processed_segments = st.session_state.processed_segments
            layout_info = st.session_state.doc_layout_info
            page_metadata = st.session_state.page_metadata
            entity_stats = st.session_state.entity_stats
            
            # Document Overview Tab
            with tabs[0]:
                show_document_overview(uploaded_file.name, processed_segments, layout_info, page_metadata)
            
            # Raw Data Tab
            with tabs[1]:
                show_raw_data(processed_segments, output_format)
            
            # Clean up the temporary file
            try:
                os.unlink(pdf_path)
            except:
                pass
    
    else:
        # Display information when no file is uploaded
        st.info("👈 Please upload a PDF document using the sidebar to get started.")
        
        # Sample output structure with collapsible details
        col1, col2 = st.columns([1, 1])
        with col1:
            st.header("Sample Output Structure")
            with st.expander("View Sample JSON Output", expanded=True):
                sample_output = {
                  "segment_level": "",
                  "segment_title": "",
                  "segment_date": "",
                  "segment_source": "",
                  "segment_text": "",
                  "start_index": "",
                  "end_index": "",
                  "named_entities": {
                    "persons": [""],
                    "organizations": [""],
                    "locations": [""],
                    "dates": [""]
                  }
                }
                st.json(sample_output)
        
        with col2:
            st.header("What This Tool Does")
            st.markdown("""
            This tool intelligently processes PDF documents to extract structured information:
            
            1. **Document Structure Recognition**:
               - Identifies hierarchical segments (chapters, sections, subsections)
               - Determines segment boundaries and relationships
            
            2. **Smart Metadata Extraction**:
               - Extracts dates, sources, and other metadata from each segment
               - Handles different date formats and writing styles
            
            3. **Advanced Entity Recognition**:
               - Identifies people, organizations, locations, and other entities
               - Uses context to disambiguate similar entities
               - Links entities to their relevant document sections
            
            4. **Visualizations & Analysis**:
               - Generates hierarchical document maps
               - Provides interactive entity analysis
               - Exports data in structured JSON format
            """)

# Helper functions for document visualization and analysis
def compute_entity_statistics(segments):
    """# Compute statistics about entities in segments for visualization (optimized)."""
    entity_stats = {
        'total_count': {},
        'by_segment': {},
        'by_level': {},
    }
    
    # Use a more efficient approach with predefined entity types
    entity_types = ["persons", "organizations", "locations", "dates", "misc"]
    
    # Initialize counters just once
    for entity_type in entity_types:
        entity_stats['total_count'][entity_type] = Counter()
        entity_stats['by_level'][entity_type] = {}
    
    # Process each segment more efficiently
    for idx, segment in enumerate(segments):
        level = segment['segment_level']
        
        # Add minimal segment info to stats
        entity_stats['by_segment'][idx] = {
            'title': segment.get('segment_title', f"Segment {idx}"),
            'level': level,
            'counts': {}
        }
        
        # Count entities by type in this segment
        for entity_type in entity_types:
            if entity_type in segment['named_entities']:
                # Use direct assignment for better performance
                entities = segment['named_entities'][entity_type]
                if not entities:
                    continue
                    
                # Limit to top 30 entities per segment for better performance
                if len(entities) > 30:
                    entities = entities[:30]
                    
                # Count entities in this segment
                segment_counter = Counter(entities)
                entity_stats['by_segment'][idx]['counts'][entity_type] = segment_counter
                
                # Add to total counts
                entity_stats['total_count'][entity_type] += segment_counter
                
                # Group by level
                if level not in entity_stats['by_level'][entity_type]:
                    entity_stats['by_level'][entity_type][level] = Counter()
                entity_stats['by_level'][entity_type][level] += segment_counter
    
    return entity_stats

def show_document_overview(filename, segments, layout_info, page_metadata):
    """Show document overview with key statistics and structure."""
    st.header(f"Document Overview: {filename}")
    
    # Document stats in metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Segments", len(segments))
    with col2:
        st.metric("Hierarchical Levels", len(set(s["segment_level"] for s in segments)))
    with col3:
        st.metric("Pages", len(page_metadata) if page_metadata else "N/A")
    with col4:
        has_columns = layout_info.get("multi_column", False) if layout_info else False
        st.metric("Layout", f"{'Multi-column' if has_columns else 'Single-column'}")
    
    # Document structure visualization
    st.subheader("Document Structure")
    
    # Create a hierarchical view of segments
    for i, segment in enumerate(segments):
        level = segment["segment_level"]
        title = segment["segment_title"]
        
        # Use CSS classes for indentation based on level
        st.markdown(f"<div class='hierarchy-level-{min(level, 5)}'>"  # Limit to 5 levels of styling
                    f"<strong>{title}</strong>"  # Bold title
                    f"</div>", unsafe_allow_html=True)
    
    # Layout information if available
    if layout_info:
        st.subheader("Document Layout Information")
        st.json(layout_info)

def show_structure_analysis(segments, layout_info):
    """Show detailed analysis of document structure and hierarchy."""
    st.header("Document Structure Analysis")
    
    # Create a tree diagram of the document structure
    st.subheader("Hierarchical Structure")
    
    # Prepare data for tree visualization
    tree_data = []
    segment_text_lengths = [len(segment["segment_text"]) for segment in segments]
    max_length = max(segment_text_lengths) if segment_text_lengths else 1
    
    for i, segment in enumerate(segments):
        relative_size = len(segment["segment_text"]) / max_length
        tree_data.append({
            "id": i,
            "parent": find_parent_segment(segments, i),
            "name": segment["segment_title"],
            "segment_level": segment["segment_level"],
            "value": len(segment["segment_text"]),
            "relative_size": relative_size
        })
    
    # Create a dataframe for visualization
    df = pd.DataFrame(tree_data)
    
    # Segment levels distribution
    level_counts = Counter([s["segment_level"] for s in segments])
    level_labels = [f"Level {level}" for level in sorted(level_counts.keys())]
    level_values = [level_counts[level] for level in sorted(level_counts.keys())]
    
    # Create a bar chart of segment levels
    fig = px.bar(
        x=level_labels,
        y=level_values,
        labels={"x": "Hierarchy Level", "y": "Number of Segments"},
        title="Distribution of Segment Hierarchy Levels",
        color=level_values,
        color_continuous_scale="Viridis"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment details in an interactive table
    st.subheader("Segment Details")
    
    # Create a dataframe for the segments table
    segments_df = pd.DataFrame([
        {
            "Segment Title": s["segment_title"],
            "Level": s["segment_level"],
            "Date": s["segment_date"] if s["segment_date"] else "N/A",
            "Source": s["segment_source"] if s["segment_source"] else "N/A",
            "Text Length": len(s["segment_text"]),
            "Entities Count": sum(len(entities) for entity_type, entities in s["named_entities"].items())
        } for s in segments
    ])
    
    st.dataframe(segments_df, use_container_width=True)

def show_entity_analysis(segments, entity_stats):
    """Show streamlined analysis of named entities for better performance."""
    st.header("Named Entity Analysis")
    
    if not entity_stats or entity_stats.get("total_count") is None:
        st.warning("No entity statistics available. Please process a document first.")
        return
    
    # Overview metrics - simplified
    entity_counts = entity_stats["total_count"]
    
    # Limit to most important entity types for better performance
    main_entity_types = ["persons", "organizations", "locations"]
    entity_types = [et for et in main_entity_types if et in entity_counts and len(entity_counts[et]) > 0]
    
    if not entity_types:
        # Get entity types with non-empty counts
        entity_types = [et for et in entity_counts.keys() if len(entity_counts[et]) > 0]
        if entity_types:
            entity_types = entity_types[:3]  # Limit to first 3 types if main types not found
    
    if not entity_types:
        st.info("No entities detected in this document.")
        return
    
    # Create metrics for each entity type using a simple row of metrics
    cols = st.columns(len(entity_types))
    for i, entity_type in enumerate(entity_types):
        with cols[i]:
            st.metric(f"{entity_type.capitalize()}", sum(entity_counts[entity_type].values()))
    
    # Top entities - simplified visualization
    st.subheader("Top Entities")
    
    # Create one tab per entity type
    entity_tabs = st.tabs([type.capitalize() for type in entity_types])
    
    for i, entity_type in enumerate(entity_types):
        with entity_tabs[i]:
            # Extract top entities from the Counter
            top_entities_counter = entity_counts[entity_type].most_common(10)
            if not top_entities_counter:
                st.info(f"No {entity_type} found in the document.")
                continue
            
            # Skip the complex visualization and go straight to dataframe
            # This is much more performance-efficient
            st.markdown(f"**Top {entity_type.capitalize()}**")
            top_entities_df = pd.DataFrame(top_entities_counter, columns=["Entity", "Count"])
            st.table(top_entities_df)  # Use table instead of dataframe for better performance
    
    # Simplified entity distribution - no heatmap
    st.subheader("Entity by Document Section")
    
    # Let user choose an entity type (but with a more efficient approach)
    selected_entity_type = st.selectbox(
        "Select entity type",
        entity_types,
        index=0
    )
    
    # Create a simpler visualization showing entity distribution
    # Instead of a complex heatmap, use a simpler bar chart
    segment_entities = {}
    
    # Only process first 10 segments and first 10 entities for performance
    for idx, segment in enumerate(segments[:10]):
        if segment.get("segment_title"):
            title = segment["segment_title"]
            if len(title) > 20:  # Truncate long titles
                title = title[:20] + "..."
                
            # Count entities in this segment
            entity_count = len(segment["named_entities"].get(selected_entity_type, []))
            if entity_count > 0:
                segment_entities[title] = entity_count
    
    if segment_entities:
        # Create a simple bar chart
        fig = px.bar(
            x=list(segment_entities.keys()),
            y=list(segment_entities.values()),
            labels={"x": "Document Section", "y": "Entity Count"},
            title=f"{selected_entity_type.capitalize()} per Document Section"
        )
        fig.update_layout(
            xaxis={'tickangle': 45},
            showlegend=False,  # Disable legend for performance
            height=400  # Set fixed height
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No {selected_entity_type} found across document sections.")

def show_raw_data(segments, output_format):
    """Show the raw extracted data with download options."""
    st.header("Raw Extracted Data")
    
    # Tabs for different output formats
    format_tabs = st.tabs(["JSON", "CSV", "Single Segment View"])
    
    # JSON Tab
    with format_tabs[0]:
        st.json(segments)
        
        # Create a download button for JSON
        output_json = json.dumps(segments, indent=2)
        st.download_button(
            label="⬇️ Download JSON Data",
            data=output_json,
            file_name="document_analysis_results.json",
            mime="application/json"
        )
    
    # CSV Tab 
    with format_tabs[1]:
        # Flatten the data for CSV
        flat_data = []
        for seg in segments:
            flat_seg = seg.copy()
            # Flatten nested structures
            for entity_type, entities in seg['named_entities'].items():
                flat_seg[f"entities_{entity_type}"] = ", ".join(entities) if entities else ""
            # Remove the nested structure
            flat_seg.pop('named_entities')
            flat_data.append(flat_seg)
        
        # Create a dataframe and display it
        df = pd.DataFrame(flat_data)
        st.dataframe(df, use_container_width=True)
        
        # Create a download button for CSV
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download CSV Data",
            data=csv_buffer.getvalue(),
            file_name="document_analysis_results.csv",
            mime="text/csv"
        )
    
    # Single Segment View
    with format_tabs[2]:
        segment_idx = st.selectbox(
            "Select a segment to view in detail",
            range(len(segments)),
            format_func=lambda i: f"{segments[i]['segment_title']} (Level {segments[i]['segment_level']})"
        )
        
        st.subheader(f"Segment: {segments[segment_idx]['segment_title']}")
        
        # Show the segment details in a formatted way
        segment = segments[segment_idx]
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"**Level:** {segment['segment_level']}")
            st.markdown(f"**Date:** {segment['segment_date'] if segment['segment_date'] else 'Not found'}")
            st.markdown(f"**Source:** {segment['source_date'] if 'source_date' in segment else 'Not found'}")
            
            st.markdown("### Named Entities")
            for entity_type, entities in segment['named_entities'].items():
                if entities:
                    st.markdown(f"**{entity_type.capitalize()}**:")
                    st.write(", ".join(entities))
        
        with col2:
            st.markdown("### Segment Text")
            st.text_area("Text content", segment['segment_text'], height=300)

def find_parent_segment(segments, segment_idx):
    """Find the parent segment index based on hierarchy level."""
    current_segment = segments[segment_idx]
    current_level = current_segment["segment_level"]
    
    # Root level segments don't have a parent
    if current_level == 1:
        return None
    
    # Look backwards for a segment with a lower level - that's the parent
    for i in range(segment_idx - 1, -1, -1):
        if segments[i]["segment_level"] < current_level:
            return i
    
    # If no parent found, return None
    return None

def show_documentation():
    """Show documentation about the system."""
    st.title("System Documentation")
    
    st.markdown("""
    ## Advanced Document Analyzer
    
    This application provides a comprehensive solution for hierarchical document segmentation and named entity recognition (NER) from PDF documents.
    
    ### Key Components
    
    #### 1. Document Reading & Layout Analysis
    - **PDF Ingestion**: Extract text with pdfplumber with layout awareness
    - **Advanced Layout Detection**: Identify multi-column layouts, headers/footers
    - **Table of Contents Detection**: Identify and process document structure hints
    - **Smart Header/Footer Removal**: Automatically identifies and removes repeating headers and footers
    
    #### 2. Hierarchical Segmentation
    - **Pattern-based Heading Detection**: Identify section titles with regex and heuristics
    - **Hierarchy Inference**: Determine the hierarchical relationships between segments
    - **Segment Boundary Detection**: Calculate precise start/end indices for segments
    - **Natural Paragraph Boundaries**: Ensures text isn't cut off mid-paragraph
    
    #### 3. Named Entity Recognition
    - **Entity Extraction**: Identify and classify named entities with spaCy
    - **Entity Disambiguation**: Resolve entity co-references and variations
    - **Enhanced Entity Types**: Extract persons, organizations, locations, dates, and more
    - **Entity Statistics**: Aggregate entity counts and distributions
    
    #### 4. Metadata Extraction
    - **Date Detection**: Extract dates from text with multiple format support
    - **Source Attribution**: Identify document and segment sources
    - **Page Mapping**: Track which pages each segment spans
    
    ### Data Format
    
    The system produces structured JSON data in this format:
    ```json
    {
      "segment_level": 1,
      "segment_title": "",
      "segment_date": "",
      "segment_source": "",
      "segment_text": "",
      "start_index": 0,
      "end_index": 0,
      "pages": [],
      "named_entities": {
        "persons": [],
        "organizations": [],
        "locations": [],
        "dates": []
      }
    }
    ```
    
    ### Technology Stack
    
    - **PDF Processing**: pdfplumber for accurate text extraction
    - **Natural Language Processing**: spaCy with large model
    - **User Interface**: Streamlit with interactive data exploration
    - **Text Analysis**: NLTK for additional linguistic processing

    """)

# show_examples function removed as it's been integrated into documentation

if __name__ == "__main__":
    main()
