"""
Pytest configuration and shared fixtures.
"""
import os
import pytest
from fastapi.testclient import TestClient
from dotenv import load_dotenv

# Load test environment variables
load_dotenv()


@pytest.fixture(scope="session")
def test_env():
    """Load test environment variables."""
    return {
        "mistral_api_key": os.getenv("Mistral_api_key"),
        "pinecone_api_key": os.getenv("Pinecone_api_key"),
        "index_name": "section-aware-rag-index-test"
    }


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    from app import app
    return TestClient(app)


@pytest.fixture
def sample_markdown_content():
    """Sample markdown content for testing uploads."""
    return """# Test Document

## Introduction
This is a test document for the RAG system.

<!-- PAGE BREAK -->

## Methods
We used advanced testing methodologies.

<!-- PAGE BREAK -->

## Results
The results were conclusive.

## Conclusion
Testing is important.
"""


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    return "What are the results of the study?"


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    class MockResponse:
        def __init__(self, content):
            self.content = content
    
    return MockResponse


@pytest.fixture
def sample_documents():
    """Sample documents for testing retrieval."""
    from langchain_core.documents import Document
    
    return [
        Document(
            page_content="This is page 1 content about AI.",
            metadata={
                "source": "test_doc.md",
                "section": "Introduction",
                "page_number": 1,
                "type": "page"
            }
        ),
        Document(
            page_content="This is page 2 content about machine learning.",
            metadata={
                "source": "test_doc.md",
                "section": "Methods",
                "page_number": 2,
                "type": "page"
            }
        )
    ]