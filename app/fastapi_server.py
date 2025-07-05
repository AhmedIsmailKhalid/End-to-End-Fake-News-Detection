import json
import time
import joblib
import logging
import hashlib
import uvicorn
import asyncio
import aiofiles
import traceback
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks, status



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/fastapi_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Rate limiting storage
rate_limit_storage = defaultdict(list)

class ModelManager:
    """Manages model loading and health checks"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.model_metadata = {}
        self.last_health_check = None
        self.health_status = "unknown"
        self.load_model()
    
    def load_model(self):
        """Load model with comprehensive error handling"""
        try:
            logger.info("Loading ML model...")
            
            # Try to load pipeline first (preferred)
            pipeline_path = Path("/tmp/model/pipeline.pkl")
            if pipeline_path.exists():
                self.pipeline = joblib.load(pipeline_path)
                self.model = self.pipeline.named_steps.get('model')
                self.vectorizer = self.pipeline.named_steps.get('vectorize')
                logger.info("Loaded model pipeline successfully")
            else:
                # Fallback to individual components
                model_path = Path("/tmp/model.pkl")
                vectorizer_path = Path("/tmp/vectorizer.pkl")
                
                if model_path.exists() and vectorizer_path.exists():
                    self.model = joblib.load(model_path)
                    self.vectorizer = joblib.load(vectorizer_path)
                    logger.info("Loaded model components successfully")
                else:
                    raise FileNotFoundError("No model files found")
            
            # Load metadata
            metadata_path = Path("/tmp/metadata.json")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
                logger.info(f"Loaded model metadata: {self.model_metadata.get('model_version', 'Unknown')}")
            
            self.health_status = "healthy"
            self.last_health_check = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.health_status = "unhealthy"
            self.model = None
            self.vectorizer = None
            self.pipeline = None
    
    def predict(self, text: str) -> tuple[str, float]:
        """Make prediction with error handling"""
        try:
            if self.pipeline:
                # Use pipeline for prediction
                prediction = self.pipeline.predict([text])[0]
                probabilities = self.pipeline.predict_proba([text])[0]
            elif self.model and self.vectorizer:
                # Use individual components
                X = self.vectorizer.transform([text])
                prediction = self.model.predict(X)[0]
                probabilities = self.model.predict_proba(X)[0]
            else:
                raise ValueError("No model available for prediction")
            
            # Get confidence score
            confidence = float(probabilities[prediction])
            
            # Convert prediction to readable format
            label = "Fake" if prediction == 1 else "Real"
            
            return label, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test prediction with sample text
            test_text = "This is a test article for health check purposes."
            label, confidence = self.predict(test_text)
            
            self.health_status = "healthy"
            self.last_health_check = datetime.now()
            
            return {
                "status": "healthy",
                "last_check": self.last_health_check.isoformat(),
                "model_available": self.model is not None,
                "vectorizer_available": self.vectorizer is not None,
                "pipeline_available": self.pipeline is not None,
                "test_prediction": {"label": label, "confidence": confidence}
            }
            
        except Exception as e:
            self.health_status = "unhealthy"
            self.last_health_check = datetime.now()
            
            return {
                "status": "unhealthy",
                "last_check": self.last_health_check.isoformat(),
                "error": str(e),
                "model_available": self.model is not None,
                "vectorizer_available": self.vectorizer is not None,
                "pipeline_available": self.pipeline is not None
            }

# Global model manager
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    logger.info("Starting FastAPI application...")
    
    # Startup tasks
    model_manager.load_model()
    
    # Schedule periodic health checks
    asyncio.create_task(periodic_health_check())
    
    yield
    
    # Shutdown tasks
    logger.info("Shutting down FastAPI application...")

# Create FastAPI app
app = FastAPI(
    title="Fake News Detection API",
    description="Production-ready API for fake news detection with comprehensive monitoring and security features",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Request/Response models
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze for fake news detection")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty')
        
        # Basic content validation
        if len(v.strip()) < 10:
            raise ValueError('Text must be at least 10 characters long')
        
        # Check for suspicious patterns
        suspicious_patterns = ['<script', 'javascript:', 'data:']
        if any(pattern in v.lower() for pattern in suspicious_patterns):
            raise ValueError('Text contains suspicious content')
        
        return v.strip()

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Prediction result: 'Real' or 'Fake'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    model_version: str = Field(..., description="Version of the model used for prediction")
    timestamp: str = Field(..., description="Timestamp of the prediction")
    processing_time: float = Field(..., description="Time taken for processing in seconds")

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=10, description="List of texts to analyze")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('Texts list cannot be empty')
        
        for text in v:
            if not text or not text.strip():
                raise ValueError('All texts must be non-empty')
            
            if len(text.strip()) < 10:
                raise ValueError('All texts must be at least 10 characters long')
        
        return [text.strip() for text in v]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_count: int
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_health: Dict[str, Any]
    system_health: Dict[str, Any]
    api_health: Dict[str, Any]

# Rate limiting
async def rate_limit_check(request: Request):
    """Check rate limits"""
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old entries
    rate_limit_storage[client_ip] = [
        timestamp for timestamp in rate_limit_storage[client_ip]
        if current_time - timestamp < 3600  # 1 hour window
    ]
    
    # Check rate limit (100 requests per hour)
    if len(rate_limit_storage[client_ip]) >= 100:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 100 requests per hour."
        )
    
    # Add current request
    rate_limit_storage[client_ip].append(current_time)

# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    log_data = {
        "method": request.method,
        "url": str(request.url),
        "client_ip": request.client.host,
        "status_code": response.status_code,
        "process_time": process_time,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"Request: {json.dumps(log_data)}")
    
    return response

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    error_data = {
        "error": True,
        "message": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat(),
        "path": request.url.path
    }
    
    logger.error(f"HTTP Exception: {json.dumps(error_data)}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_data
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    error_data = {
        "error": True,
        "message": "Internal server error",
        "timestamp": datetime.now().isoformat(),
        "path": request.url.path
    }
    
    logger.error(f"General Exception: {str(exc)}\n{traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content=error_data
    )

# Background tasks
async def periodic_health_check():
    """Periodic health check"""
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            health_status = model_manager.health_check()
            
            if health_status["status"] == "unhealthy":
                logger.warning("Model health check failed, attempting to reload...")
                model_manager.load_model()
                
        except Exception as e:
            logger.error(f"Periodic health check failed: {e}")

# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Fake News Detection API",
        "version": "2.0.0",
        "documentation": "/docs",
        "health_check": "/health"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    _: None = Depends(rate_limit_check)
):
    """
    Predict whether a news article is fake or real
    
    - **text**: The news article text to analyze
    - **returns**: Prediction result with confidence score
    """
    start_time = time.time()
    
    try:
        # Check model health
        if model_manager.health_status != "healthy":
            raise HTTPException(
                status_code=503,
                detail="Model is not available. Please try again later."
            )
        
        # Make prediction
        label, confidence = model_manager.predict(request.text)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create response
        response = PredictionResponse(
            prediction=label,
            confidence=confidence,
            model_version=model_manager.model_metadata.get('model_version', 'unknown'),
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
        # Log prediction (background task)
        background_tasks.add_task(
            log_prediction,
            request.text,
            label,
            confidence,
            http_request.client.host,
            processing_time
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    _: None = Depends(rate_limit_check)
):
    """
    Predict multiple news articles in batch
    
    - **texts**: List of news article texts to analyze
    - **returns**: List of prediction results
    """
    start_time = time.time()
    
    try:
        # Check model health
        if model_manager.health_status != "healthy":
            raise HTTPException(
                status_code=503,
                detail="Model is not available. Please try again later."
            )
        
        predictions = []
        
        for text in request.texts:
            try:
                label, confidence = model_manager.predict(text)
                
                prediction = PredictionResponse(
                    prediction=label,
                    confidence=confidence,
                    model_version=model_manager.model_metadata.get('model_version', 'unknown'),
                    timestamp=datetime.now().isoformat(),
                    processing_time=0.0  # Will be updated with total time
                )
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Batch prediction failed for text: {e}")
                # Continue with other texts
                continue
        
        # Calculate total processing time
        total_processing_time = time.time() - start_time
        
        # Update processing time for all predictions
        for prediction in predictions:
            prediction.processing_time = total_processing_time / len(predictions)
        
        response = BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            processing_time=total_processing_time
        )
        
        # Log batch prediction (background task)
        background_tasks.add_task(
            log_batch_prediction,
            len(request.texts),
            len(predictions),
            http_request.client.host,
            total_processing_time
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Comprehensive health check endpoint
    
    - **returns**: Detailed health status of the API and model
    """
    try:
        # Model health
        model_health = model_manager.health_check()
        
        # System health
        import psutil
        system_health = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "uptime": time.time() - psutil.boot_time()
        }
        
        # API health
        api_health = {
            "rate_limit_active": len(rate_limit_storage) > 0,
            "active_connections": len(rate_limit_storage)
        }
        
        # Overall status
        overall_status = "healthy" if model_health["status"] == "healthy" else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now().isoformat(),
            model_health=model_health,
            system_health=system_health,
            api_health=api_health
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            model_health={"status": "unhealthy", "error": str(e)},
            system_health={"error": str(e)},
            api_health={"error": str(e)}
        )

@app.get("/metrics")
async def get_metrics():
    """
    Get API metrics
    
    - **returns**: Usage statistics and performance metrics
    """
    try:
        # Calculate metrics from rate limiting storage
        total_requests = sum(len(requests) for requests in rate_limit_storage.values())
        unique_clients = len(rate_limit_storage)
        
        metrics = {
            "total_requests": total_requests,
            "unique_clients": unique_clients,
            "model_version": model_manager.model_metadata.get('model_version', 'unknown'),
            "model_health": model_manager.health_status,
            "last_health_check": model_manager.last_health_check.isoformat() if model_manager.last_health_check else None,
            "timestamp": datetime.now().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Metrics retrieval failed: {str(e)}"
        )

@app.post("/model/reload")
async def reload_model():
    """
    Reload the ML model
    
    - **returns**: Status of model reload operation
    """
    try:
        logger.info("Manual model reload requested")
        model_manager.load_model()
        
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_health": model_manager.health_status,
            "model_version": model_manager.model_metadata.get('model_version', 'unknown'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model reload failed: {str(e)}"
        )

# Background task functions
async def log_prediction(text: str, prediction: str, confidence: float, client_ip: str, processing_time: float):
    """Log prediction details"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "client_ip": client_ip,
            "text_length": len(text),
            "prediction": prediction,
            "confidence": confidence,
            "processing_time": processing_time,
            "text_hash": hashlib.md5(text.encode()).hexdigest()
        }
        
        # Save to log file
        log_file = Path("/tmp/prediction_log.json")
        
        # Load existing logs
        logs = []
        if log_file.exists():
            try:
                async with aiofiles.open(log_file, 'r') as f:
                    content = await f.read()
                    logs = json.loads(content)
            except:
                logs = []
        
        # Add new log
        logs.append(log_entry)
        
        # Keep only last 1000 entries
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        # Save logs
        async with aiofiles.open(log_file, 'w') as f:
            await f.write(json.dumps(logs, indent=2))
            
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")

async def log_batch_prediction(total_texts: int, successful_predictions: int, client_ip: str, processing_time: float):
    """Log batch prediction details"""
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "batch_prediction",
            "client_ip": client_ip,
            "total_texts": total_texts,
            "successful_predictions": successful_predictions,
            "processing_time": processing_time,
            "success_rate": successful_predictions / total_texts if total_texts > 0 else 0
        }
        
        logger.info(f"Batch prediction logged: {json.dumps(log_entry)}")
        
    except Exception as e:
        logger.error(f"Failed to log batch prediction: {e}")

# Custom OpenAPI
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Fake News Detection API",
        version="2.0.0",
        description="Production-ready API for fake news detection with comprehensive monitoring and security features",
        routes=app.routes,
    )
    
    # Add security definitions
    openapi_schema["components"]["securitySchemes"] = {
        "Bearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_server:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        reload=False,
        access_log=True
    )