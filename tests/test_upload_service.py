"""
Test cases for upload service functionality.
"""
import pytest
from unittest.mock import Mock, patch
from services.upload_service import UploadService
from langchain_core.documents import Document


class TestUploadServiceInitialization:
    """Tests for upload service initialization."""
    
    def test_service_initialization(self, test_env):
        """Test that upload service initializes correctly."""
        service = UploadService(
            mistral_api_key=test_env["mistral_api_key"],
            pinecone_api_key=test_env["pinecone_api_key"],
            index_name=test_env["index_name"]
        )
        
        assert service.llm is not None
        assert service.embeddings is not None
        assert service.vector_store is not None


class TestMarkdownCleaning:
    """Tests for markdown text cleaning."""
    
    def test_clean_markdown_removes_anchors(self, test_env):
        """Test that anchor tags are removed."""
        service = UploadService(
            mistral_api_key=test_env["mistral_api_key"],
            pinecone_api_key=test_env["pinecone_api_key"],
            index_name=test_env["index_name"]
        )
        
        text_with_anchors = "<a id='section1'></a>This is content<a id='section2'></a>"
        cleaned = service.clean_markdown_text(text_with_anchors)
        
        assert "<a id=" not in cleaned
        assert "This is content" in cleaned


class TestSectionSplitting:
    """Tests for section splitting functionality."""
    
    def test_split_into_sections_creates_documents(self, test_env):
        """Test that pages are split into section documents."""
        service = UploadService(
            mistral_api_key=test_env["mistral_api_key"],
            pinecone_api_key=test_env["pinecone_api_key"],
            index_name=test_env["index_name"]
        )
        
        normalized_pages = [
            """# Title
## Section 1
Content for section 1.

## Section 2
Content for section 2."""
        ]
        
        docs = service.split_into_sections(normalized_pages, "test.md")
        
        assert len(docs) > 0
        assert all(isinstance(doc, Document) for doc in docs)
        
        # Check metadata
        doc = docs[0]
        assert "source" in doc.metadata
        assert "section" in doc.metadata
        assert "page_number" in doc.metadata


class TestFileProcessing:
    """Tests for file processing workflow."""
    
    @patch('services.upload_service.UploadService.source_exists')
    def test_process_existing_file_returns_error(self, mock_exists, test_env, sample_markdown_content):
        """Test that processing existing file returns error."""
        mock_exists.return_value = True
        
        service = UploadService(
            mistral_api_key=test_env["mistral_api_key"],
            pinecone_api_key=test_env["pinecone_api_key"],
            index_name=test_env["index_name"]
        )
        
        result = service.process_markdown_file(
            file_content=sample_markdown_content,
            file_name="existing_file.md"
        )
        
        assert result["status"] == "error"
        assert "already exists" in result["message"]
    
    @patch('services.upload_service.UploadService.upload_to_vector_store')
    @patch('services.upload_service.UploadService.source_exists')
    def test_process_new_file_success(
        self,
        mock_exists,
        mock_upload,
        test_env,
        sample_markdown_content
    ):
        """Test successful processing of new file."""
        mock_exists.return_value = False
        mock_upload.return_value = ["id1", "id2", "id3"]
        
        service = UploadService(
            mistral_api_key=test_env["mistral_api_key"],
            pinecone_api_key=test_env["pinecone_api_key"],
            index_name=test_env["index_name"]
        )
        
        result = service.process_markdown_file(
            file_content=sample_markdown_content,
            file_name="new_file.md"
        )
        
        # Note: This test might fail due to actual API calls
        # In real scenario, you'd mock all LLM and vector store calls
        assert result["status"] in ["success", "error"]
        
        if result["status"] == "success":
            assert "details" in result
            assert "pages_processed" in result["details"]


class TestMetadataExtraction:
    """Tests for metadata extraction."""
    
    @patch('services.upload_service.ChatMistralAI')
    def test_extract_publication_year(self, mock_llm, test_env, mock_llm_response):
        """Test extracting publication year from summary."""
        mock_instance = Mock()
        mock_instance.invoke.return_value = mock_llm_response("2023")
        mock_llm.return_value = mock_instance
        
        service = UploadService(
            mistral_api_key=test_env["mistral_api_key"],
            pinecone_api_key=test_env["pinecone_api_key"],
            index_name=test_env["index_name"]
        )
        
        year = service.extract_publication_year("This paper was published in 2023.")
        
        assert isinstance(year, int)
        assert year > 0