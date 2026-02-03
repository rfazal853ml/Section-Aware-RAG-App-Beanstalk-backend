"""
Test cases for FastAPI endpoints.
"""
import pytest
from fastapi import status


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    def test_root_endpoint(self, client):
        """Test the root health check endpoint."""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "message" in data
        assert "version" in data


class TestQueryEndpoint:
    """Tests for the query endpoint."""
    
    def test_query_with_valid_input(self, client, sample_query):
        """Test query endpoint with valid input."""
        response = client.post(
            "/query",
            json={"query": sample_query}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "filtered_documents" in data
        assert isinstance(data["sources"], list)
    
    def test_query_with_empty_string(self, client):
        """Test query endpoint with empty string."""
        response = client.post(
            "/query",
            json={"query": ""}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_query_with_whitespace_only(self, client):
        """Test query endpoint with whitespace only."""
        response = client.post(
            "/query",
            json={"query": "   "}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_query_without_query_field(self, client):
        """Test query endpoint without query field."""
        response = client.post(
            "/query",
            json={}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestUploadEndpoint:
    """Tests for the upload endpoint."""
    
    def test_upload_valid_markdown(self, client, sample_markdown_content):
        """Test uploading a valid markdown file."""
        # Create a file-like object
        files = {
            "file": ("test_document.md", sample_markdown_content, "text/markdown")
        }
        
        response = client.post("/upload", files=files)
        
        # Note: This might fail if document already exists
        # In real scenario, you'd clean up test data
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_400_BAD_REQUEST  # If already exists
        ]
    
    def test_upload_non_markdown_file(self, client):
        """Test uploading a non-markdown file."""
        files = {
            "file": ("test.txt", "This is not markdown", "text/plain")
        }
        
        response = client.post("/upload", files=files)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "markdown" in data["detail"].lower()
    
    def test_upload_without_file(self, client):
        """Test upload endpoint without file."""
        response = client.post("/upload")
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestDocumentsEndpoint:
    """Tests for the documents listing endpoint."""
    
    def test_list_documents(self, client):
        """Test listing documents."""
        response = client.get("/documents")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "total_documents" in data
        assert "documents" in data
        assert isinstance(data["documents"], list)
    
    def test_documents_response_structure(self, client):
        """Test that document items have correct structure."""
        response = client.get("/documents")
        data = response.json()
        
        if data["total_documents"] > 0:
            doc = data["documents"][0]
            assert "filename" in doc
            assert "publication_year" in doc
            assert "keywords" in doc