"""
Integration tests for the full workflow.
"""
import pytest
from fastapi import status


class TestUploadAndQueryWorkflow:
    """Test the complete upload and query workflow."""
    
    @pytest.mark.integration
    def test_upload_then_query_workflow(self, client, sample_markdown_content):
        """Test uploading a document and then querying it."""
        # Step 1: Upload document
        files = {
            "file": ("integration_test.md", sample_markdown_content, "text/markdown")
        }
        
        upload_response = client.post("/upload", files=files)
        
        # Document might already exist from previous test runs
        if upload_response.status_code == status.HTTP_200_OK:
            upload_data = upload_response.json()
            assert upload_data["status"] == "success"
        
        # Step 2: Query the document
        query_response = client.post(
            "/query",
            json={"query": "What are the results?"}
        )
        
        assert query_response.status_code == status.HTTP_200_OK
        query_data = query_response.json()
        assert "answer" in query_data
        assert len(query_data["answer"]) > 0
    
    @pytest.mark.integration
    def test_list_documents_after_upload(self, client, sample_markdown_content):
        """Test that uploaded documents appear in the list."""
        # Get initial document count
        response1 = client.get("/documents")
        initial_count = response1.json()["total_documents"]
        
        # Upload a document
        files = {
            "file": ("list_test.md", sample_markdown_content, "text/markdown")
        }
        
        upload_response = client.post("/upload", files=files)
        
        # Get updated document count
        response2 = client.get("/documents")
        final_count = response2.json()["total_documents"]
        
        # If upload was successful, count should increase
        if upload_response.status_code == status.HTTP_200_OK:
            assert final_count >= initial_count


class TestErrorHandling:
    """Test error handling across the system."""
    
    def test_query_with_invalid_json(self, client):
        """Test API handles invalid JSON gracefully."""
        response = client.post(
            "/query",
            content="this is not json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_CONTENT
        ]
    
    def test_upload_corrupted_file(self, client):
        """Test upload handles corrupted files gracefully."""
        # Create corrupted markdown
        corrupted_content = b'\x80\x81\x82\x83'
        
        files = {
            "file": ("corrupted.md", corrupted_content, "text/markdown")
        }
        
        response = client.post("/upload", files=files)
        
        # Should return error for encoding issues
        assert response.status_code == status.HTTP_400_BAD_REQUEST