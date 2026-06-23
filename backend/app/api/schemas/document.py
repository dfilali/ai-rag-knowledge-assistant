from pydantic import BaseModel, Field

class DocumentInfo(BaseModel):
    filename: str = Field(..., description="Name of the PDF file.")
    size_bytes: int = Field(..., description="File size in bytes.")
    created_at: float = Field(..., description="Unix timestamp of when the file was uploaded.")

class DocumentUploadResponse(BaseModel):
    filename: str = Field(..., description="Successfully uploaded filename.")
    chunks_count: int = Field(..., description="Number of text chunks created and indexed.")
    status: str = Field("success", description="Status code of the operation.")
    message: str = Field(..., description="Operation summary message.")

class DocumentDeleteResponse(BaseModel):
    status: str = Field("success", description="Status code of the operation.")
    message: str = Field(..., description="Operation summary message.")
