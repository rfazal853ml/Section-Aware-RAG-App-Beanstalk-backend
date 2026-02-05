import os
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv
from services.rag_service import RAGService
from services.upload_service import UploadService
from loguru import logger
import sys

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <green>{level: <8}</green> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Research Assistant API",
    description="API for querying documents and uploading markdown files",
    version="1.0.0"
)

# CORS middleware (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
MISTRAL_API_KEY = os.environ.get("Mistral_api_key")
PINECONE_API_KEY = os.environ.get("Pinecone_api_key")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "unstructdocwithmetadata")

if not MISTRAL_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing required API keys in environment variables")

rag_service = RAGService(
    mistral_api_key=MISTRAL_API_KEY,
    pinecone_api_key=PINECONE_API_KEY,
    index_name=INDEX_NAME
)

upload_service = UploadService(
    mistral_api_key=MISTRAL_API_KEY,
    pinecone_api_key=PINECONE_API_KEY,
    index_name=INDEX_NAME
)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Describe the Mixed-Integer Second Order Cone Programming model presented in the paper and what specific network it was tested on."
            }
        },
        extra="forbid" # Optional: strictly prevents extra fields in requests
    )
    query: str


class Source(BaseModel):
    source: str
    section: str
    page_number: int


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    filtered_documents: List[str]


class UploadResponse(BaseModel):
    status: str
    message: str
    details: Dict[str, Any] = None


# API Endpoints
@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    logger.info("Health check endpoint accessed")
    return {
        "message": "RAG Research Assistant API is running",
        "status": "healthy",
        "version": "1.0.0"
    }


@app.post(
    "/query",
    # response_model=QueryResponse,
    # tags=["Query"],
    # summary="Query the RAG system",
    # description="Submit a question to the RAG system and receive an answer with sources"
)
async def query_documents(request: QueryRequest):
    """
    Query the RAG system with a user question.
    
    - **query**: The question to ask about the documents
    
    Returns the answer along with source citations.
    """
    logger.info(f"Received query request: {request.query[:50]}...")
    if not request.query or request.query.strip() == "":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    try:
        result = rag_service.query(request.query)
        logger.info(f"Query processed successfully. Found {len(result['sources'])} sources")
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Upload"],
    summary="Upload a markdown file",
    description="Upload and process a markdown (.md) file into the vector database"
)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a markdown file for processing.
    
    - **file**: Markdown file (.md) to upload
    
    The file will be:
    1. Normalized and structured
    2. Split into sections
    3. Summarized
    4. Embedded and stored in Pinecone
    
    Returns processing status and details.
    """
    logger.info(f"Received file upload: {file.filename}")
    # Validate file type
    if not file.filename.endswith('.md'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .md (markdown) files are supported"
        )
    logger.debug(f"File validated: {file.filename}, size: {file.size if hasattr(file, 'size') else 'unknown'}")
    
    try:
        # Read file content
        content = await file.read()
        file_content = content.decode('utf-8')
        
        logger.info(f"Starting processing for: {file.filename}")
        # Process the file
        result = upload_service.process_markdown_file(
            file_content=file_content,
            file_name=file.filename
        )
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
        else:
            logger.success(f"File processed successfully: {file.filename}")
        
        return UploadResponse(**result)
    
    except UnicodeDecodeError:
        logger.error(f"Encoding error for file: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File encoding error. Please ensure the file is UTF-8 encoded."
        )
    except Exception as e:
        logger.error(f"Processing error for {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )


@app.get(
    "/documents",
    tags=["Documents"],
    summary="List uploaded documents",
    description="Get a list of all documents in the vector database"
)
async def list_documents():
    """
    Retrieve a list of all documents currently in the database.
    
    Note: This is a basic implementation. For production, consider
    implementing pagination and filtering.
    """
    logger.info("Listing documents request received")
    try:
        # Query for summary documents to get list of sources
        results = rag_service.vector_store.similarity_search(
            query="",
            k=100,
            filter={"type": "summary"}
        )
        
        documents = []
        seen_sources = set()
        
        for doc in results:
            source = doc.metadata.get('source', 'Unknown')
            if source not in seen_sources:
                documents.append({
                    "filename": source,
                    "publication_year": doc.metadata.get('publication_year', 0),
                    "keywords": doc.metadata.get('keywords', ''),
                })
                seen_sources.add(source)

        logger.info(f"Found {len(documents)} unique documents")
        return {
            "total_documents": len(documents),
            "documents": documents
        }
    
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving documents: {str(e)}"
        )

# GitHub action triggerss
if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
