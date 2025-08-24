"""
A service module for extracting text and tabular data from financial PDFs.
Uses pypdf for text and tabula-py for tables.
Saves output to the 'output' directory for quality review.
"""

import logging
from pathlib import Path
from typing import List
import pandas as pd
from pypdf import PdfReader
from tabula import read_pdf

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class PDFExtractor:
    """
    A class to extract text and tables from a PDF document.
    Designed for financial reports like 10-Qs.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        # Create an output directory named after the PDF
        self.output_dir = Path("output") / self.pdf_path.stem
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.reader = PdfReader(str(self.pdf_path))
        logger.info(f"Initialized PDFExtractor for {self.pdf_path.name}")

    def extract_text(self) -> str:
        """Extract all text from the PDF."""
        text = ""
        for i, page in enumerate(self.reader.pages):
            extracted = page.extract_text()
            if extracted:
                text += f"\n\n--- PAGE {i+1} ---\n\n{extracted}"
        logger.info("Text extraction completed.")
        return text

    def extract_tables(self, **kwargs) -> List[pd.DataFrame]:
        """Extract all tables from the PDF using tabula-py."""
        default_kwargs = {
            "pages": "all",
            "multiple_tables": True,
            "pandas_options": {"header": None}
        }
        default_kwargs.update(kwargs)

        try:
            tables: List[pd.DataFrame] = read_pdf(str(self.pdf_path), **default_kwargs)
            logger.info(f"Table extraction completed. Found {len(tables)} tables.")
            return tables
        except Exception as e:
            logger.error(f"Error extracting tables: {e}")
            return []

    def _estimate_page_for_table(self, table_index: int, total_tables: int) -> int:
        """Roughly estimate the page number for a table."""
        pages_per_table = len(self.reader.pages) / max(1, total_tables)
        return int((table_index + 1) * pages_per_table)

    def _find_table_context(self, full_text: str, table_index: int, total_tables: int) -> str:
        """Find the surrounding text context for a table."""
        pages = full_text.split("\n\n--- PAGE ")
        estimated_page_num = self._estimate_page_for_table(table_index, total_tables)
        
        if 0 < estimated_page_num < len(pages):
            # Return the first ~500 characters of the page
            return pages[estimated_page_num].strip()[:500] + "..."
        return "Context could not be determined."

    def save_extraction(self, text: str, tables: List[pd.DataFrame]):
        """Save the extracted text and enriched tables to disk."""
        # Save the full text
        text_file = self.output_dir / f"{self.pdf_path.stem}_full_text.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Full text saved to {text_file}")

        # Save each table with its rich context
        for i, table in enumerate(tables):
            context_snippet = self._find_table_context(text, i, len(tables))
            rich_table_content = (
                f"**Document:** {self.pdf_path.name}\n"
                f"**Report Period:** For the quarterly period ended June 25, 2022\n"
                f"**Table ID:** table_{i+1:02d}\n"
                f"**Estimated Location in Document:** Page {self._estimate_page_for_table(i, len(tables))}\n"
                f"**Context from Surrounding Text:**\n{context_snippet}\n\n"
                f"**Table Data (in Markdown format):**\n"
                f"{table.to_markdown(index=False) if hasattr(table, 'to_markdown') else str(table)}"
            )

            table_file = self.output_dir / f"table_{i+1:02d}_with_context.txt"
            with open(table_file, 'w', encoding='utf-8') as f:
                f.write(rich_table_content)
            logger.info(f"Table {i+1} with context saved to {table_file}")

        # Create a summary report
        summary_file = self.output_dir / "extraction_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Extraction Summary for {self.pdf_path.name}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total pages in PDF: {len(self.reader.pages)}\n")
            f.write(f"Total tables extracted: {len(tables)}\n\n")
            f.write("Table Previews:\n")
            f.write("-"*30 + "\n")
            for i, table in enumerate(tables):
                f.write(f"Table {i+1} ({table.shape[0]} rows x {table.shape[1]} columns):\n")
                f.write(table.head().to_string())
                f.write("\n\n" + "-"*30 + "\n")
        logger.info(f"Extraction summary saved to {summary_file}")

    def extract(self) -> tuple:
        """Perform a complete extraction."""
        text = self.extract_text()
        tables = self.extract_tables()
        return text, tables

# Example usage
if __name__ == "__main__":
    # Update this path to point to your PDF
    PDF_PATH = "D:/Projects/sds_rag_v3/data/2022 Q3 AAPL.pdf"

    extractor = PDFExtractor(PDF_PATH)
    full_text, table_list = extractor.extract()
    extractor.save_extraction(full_text, table_list)

    print("Extraction complete!")
    print(f"Output saved to: {extractor.output_dir}")
    print(f"Extracted {len(table_list)} tables.")
    print("First 500 characters of text:")
    print(full_text[:500] + "...")