# PDF Document Extractor UI
 poetry run python run_app.py
A simple Streamlit web interface for uploading PDF documents and extracting text/tables for RAG processing.

## 🚀 Quick Start

### Launch the UI
```bash
# Option 1: Using the launcher script
poetry run python run_app.py

# Option 2: Direct Streamlit command
poetry run streamlit run src/sds_rag/ui/streamlit_app.py
```

The app will open in your browser at: `http://localhost:8501`

## 📋 Features

### Upload & Configuration
- **File Upload**: Drag & drop or browse for PDF files
- **Extraction Options**: Toggle text and table extraction
- **Page Selection**: Extract from all pages, first 10, or custom range
- **Real-time Progress**: Progress bar and status updates

### Processing
- **Text Extraction**: Full document text with page markers
- **Table Extraction**: Tabular data using tabula-py with context
- **Error Handling**: Graceful error handling with user feedback
- **Temporary Files**: Secure handling of uploaded documents

### Results & Downloads
- **Results Summary**: Page count, tables found, text length
- **Preview**: Text and table previews directly in the UI
- **Download Options**: Individual files for text, tables, and summary
- **Output Management**: Organized output directory structure

## 🖥️ UI Layout

### Main Interface
- **Left Panel**: File upload and extraction controls
- **Right Panel**: Instructions and help information
- **Sidebar**: Configuration options and settings

### Results Display
- **Metrics**: Key extraction statistics
- **Previews**: Sample text and table content
- **Downloads**: Individual file download buttons
- **Output Location**: Path to saved extraction files

## 📁 Output Structure

```
extracted_documents/
└── {document_name}/
    ├── {document_name}_full_text.txt
    ├── extraction_summary.txt
    ├── table_01_with_context.txt
    ├── table_02_with_context.txt
    └── ...
```

## 🔧 Configuration Options

### Extraction Settings
- **Extract Text**: Toggle narrative text extraction
- **Extract Tables**: Toggle tabular data extraction
- **Page Range**: Control which pages to process

### Table Settings
- **All Pages**: Process entire document (default)
- **First 10**: Process first 10 pages only
- **Custom**: Specify custom page ranges (e.g., "1,3-5,10")

## 📊 Supported Documents

- Financial reports (10-Q, 10-K, earnings)
- Annual reports and presentations
- Research papers with tables
- Any PDF with structured content

## 🎯 RAG Integration Ready

The extracted data is formatted for immediate use in RAG pipelines:

- **Rich Context**: Tables include surrounding text context
- **Metadata**: Document source, page numbers, table IDs
- **Clean Format**: Markdown tables and structured text
- **Semantic Units**: Each table is a self-contained chunk

## 🐛 Troubleshooting

### Common Issues
1. **Java Required**: tabula-py requires Java runtime
2. **Large Files**: May take time for complex documents
3. **Unicode**: Some special characters may display incorrectly in console
4. **Permissions**: Ensure write access to output directory

### Error Messages
- Check the Streamlit interface for detailed error messages
- Verify PDF is not password protected
- Ensure sufficient disk space for extraction

## 🔄 Next Steps

After extraction, use the output files for:
1. **Vector Database**: Load into Qdrant for similarity search
2. **Embeddings**: Generate with sentence-transformers
3. **RAG Pipeline**: Integrate with LangChain for Q&A
4. **Analysis**: Process extracted data for insights