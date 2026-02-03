import re
import os
from pathlib import Path
from typing import List, Dict, Any
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from prompts import (
    normalization_prompt,
    page_summary_prompt,
    document_summary_prompt,
    publication_year_prompt,
    keywords_prompt
)


class UploadService:
    """Service for processing and uploading markdown documents."""
    
    SECTION_RE = re.compile(r"^#{1,3}\s+(.+)$")
    
    def __init__(
        self,
        mistral_api_key: str,
        pinecone_api_key: str,
        index_name: str
    ):
        """Initialize the upload service."""
        self.llm = ChatMistralAI(
            model="mistral-large-latest",
            temperature=0,
            api_key=mistral_api_key
        )
        
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            api_key=mistral_api_key
        )
        
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        self.vector_store = PineconeVectorStore(
            embedding=self.embeddings,
            index=index
        )
    
    def source_exists(self, source_name: str) -> bool:
        """Check if source already exists in the vector store."""
        try:
            results = self.vector_store.similarity_search(
                query="test",
                k=1,
                filter={"source": {"$eq": source_name}}
            )
            return len(results) > 0
        except Exception as e:
            print(f"Error checking source: {e}")
            return False
    
    def clean_markdown_text(self, text: str) -> str:
        """Clean markdown text by removing anchor tags."""
        clean_text = re.sub(r"<a id='[^']+'></a>", "", text)
        return clean_text.strip()
    
    def normalize_pages(self, pages: List[str]) -> List[str]:
        """Normalize pages using LLM."""
        normalized_pages = []
        
        for page_no, page in enumerate(pages, start=1):
            response = self.llm.invoke(
                normalization_prompt.format(page=page)
            )
            normalized_page = response.content
            normalized_pages.append(normalized_page)
            print(f"Page {page_no}/{len(pages)} normalized")
        
        return normalized_pages
    
    def split_into_sections(self, normalized_pages: List[str], file_name: str) -> List[Document]:
        """Split normalized pages into sections."""
        docs = []
        
        for page_no, page in enumerate(normalized_pages, start=1):
            current_section = None
            buffer = []
            
            for line in page.splitlines():
                line = line.strip()
                
                m = self.SECTION_RE.match(line)
                if m:
                    # Flush previous buffer
                    if current_section and buffer:
                        docs.append(
                            Document(
                                page_content=f"{current_section}\n" + "\n".join(buffer),
                                metadata={
                                    "source": file_name,
                                    "section": current_section,
                                    "page_number": page_no
                                },
                            )
                        )
                        buffer = []
                    
                    current_section = m.group(1)
                    continue
                
                buffer.append(line)
            
            # Flush last section
            if current_section and buffer:
                docs.append(
                    Document(
                        page_content=f"{current_section}\n" + "\n".join(buffer),
                        metadata={
                            "source": file_name,
                            "section": current_section,
                            "page_number": page_no
                        },
                    )
                )
        
        return docs
    
    def generate_page_summaries(self, docs: List[Document]) -> List[str]:
        """Generate summaries for each page."""
        page_summaries = []
        
        for doc in docs:
            response = self.llm.invoke(
                page_summary_prompt.format(page_content=doc.page_content)
            )
            page_summaries.append(response.content.strip())
        
        return page_summaries
    
    def generate_document_summary(self, page_summaries: List[str]) -> str:
        """Generate overall document summary."""
        response = self.llm.invoke(
            document_summary_prompt.format(
                summaries="\n".join(page_summaries)
            )
        )
        return response.content.strip()
    
    def extract_publication_year(self, document_summary: str) -> int:
        """Extract publication year from summary."""
        response = self.llm.invoke(
            publication_year_prompt.format(summary=document_summary)
        )
        
        year_str = "".join(filter(str.isdigit, response.content))
        return int(year_str) if year_str else 0
    
    def extract_keywords(self, document_summary: str) -> str:
        """Extract keywords from summary."""
        response = self.llm.invoke(
            keywords_prompt.format(summary=document_summary)
        )
        return response.content.strip()
    
    def create_final_documents(
        self,
        docs: List[Document],
        file_name: str,
        publication_year: int,
        keywords: str
    ) -> List[Document]:
        """Add metadata to documents."""
        final_docs = []
        
        for doc in docs:
            final_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        "source": file_name,
                        "section": doc.metadata.get("section", ""),
                        "page_number": int(doc.metadata.get("page_number", 0)),
                        "publication_year": int(publication_year),
                        "keywords": keywords,
                        "type": "page",
                    }
                )
            )
        
        return final_docs
    
    def split_documents(self, final_docs: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        return text_splitter.split_documents(final_docs)
    
    def upload_to_vector_store(self, documents: List[Document]) -> List[str]:
        """Upload documents to Pinecone."""
        document_ids = self.vector_store.add_documents(documents=documents)
        return document_ids
    
    def process_markdown_file(
        self,
        file_content: str,
        file_name: str
    ) -> Dict[str, Any]:
        """
        Main processing pipeline for markdown files.
        
        Args:
            file_content: Content of the markdown file
            file_name: Name of the file
        
        Returns:
            Dictionary with processing results
        """
        try:
            # Check if already exists
            if self.source_exists(file_name):
                return {
                    "status": "error",
                    "message": f"Document '{file_name}' already exists in the database."
                }
            
            # Step 1: Clean text
            clean_text = self.clean_markdown_text(file_content)
            
            # Step 2: Split into pages
            pages = clean_text.split("<!-- PAGE BREAK -->")
            
            # Step 3: Normalize pages
            normalized_pages = self.normalize_pages(pages)
            
            # Step 4: Split into sections
            docs = self.split_into_sections(normalized_pages, file_name)
            
            # Step 5: Generate page summaries
            page_summaries = self.generate_page_summaries(docs)
            
            # Step 6: Generate document summary
            document_summary = self.generate_document_summary(page_summaries)
            
            # Step 7: Extract metadata
            publication_year = self.extract_publication_year(document_summary)
            keywords = self.extract_keywords(document_summary)
            
            # Step 8: Create final documents
            final_docs = self.create_final_documents(
                docs, file_name, publication_year, keywords
            )
            
            # Step 9: Split into chunks
            split_docs = self.split_documents(final_docs)
            
            # Step 10: Add summary document
            summary_doc = Document(
                page_content=document_summary,
                metadata={
                    "source": file_name,
                    "section": "",
                    "page_number": 0,
                    "publication_year": int(publication_year),
                    "keywords": keywords,
                    "type": "summary",
                }
            )
            split_docs.append(summary_doc)
            
            # Step 11: Upload to vector store
            document_ids = self.upload_to_vector_store(split_docs)
            
            return {
                "status": "success",
                "message": f"Successfully processed '{file_name}'",
                "details": {
                    "pages_processed": len(pages),
                    "sections_created": len(docs),
                    "chunks_created": len(split_docs) - 1,
                    "document_ids_count": len(document_ids),
                    "publication_year": publication_year,
                    "keywords": keywords
                }
            }
        
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing file: {str(e)}"
            }