import os
import io
import sys
import json
import time
import hashlib
import logging
import requests
import subprocess
import pandas as pd
import altair as alt
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add root to sys.path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

class StreamlitAppManager:
    """Manages Streamlit application state and functionality"""
    
    def __init__(self):
        self.setup_config()
        self.setup_paths()
        self.setup_api_client()
        self.initialize_session_state()
    
    def setup_config(self):
        """Setup application configuration"""
        self.config = {
            'api_url': "http://localhost:8000",
            'max_upload_size': 10 * 1024 * 1024,  # 10MB
            'supported_file_types': ['csv', 'txt', 'json'],
            'max_text_length': 10000,
            'prediction_timeout': 30,
            'refresh_interval': 60,
            'max_batch_size': 10
        }
    
    def setup_paths(self):
        """Setup file paths"""
        self.paths = {
            'custom_data': Path("/tmp/custom_upload.csv"),
            'metadata': Path("/tmp/metadata.json"),
            'activity_log': Path("/tmp/activity_log.json"),
            'drift_log': Path("/tmp/logs/monitoring_log.json"),
            'prediction_log': Path("/tmp/prediction_log.json"),
            'scheduler_log': Path("/tmp/logs/scheduler_execution.json"),
            'error_log': Path("/tmp/logs/scheduler_errors.json")
        }
    
    def setup_api_client(self):
        """Setup API client with error handling"""
        self.session = requests.Session()
        self.session.timeout = self.config['prediction_timeout']
        
        # Test API connection
        self.api_available = self.test_api_connection()
    
    def test_api_connection(self) -> bool:
        """Test API connection"""
        try:
            response = self.session.get(f"{self.config['api_url']}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        
        if 'upload_history' not in st.session_state:
            st.session_state.upload_history = []
        
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False

# Initialize app manager
app_manager = StreamlitAppManager()

# Page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def load_json_file(file_path: Path, default: Any = None) -> Any:
    """Safely load JSON file with error handling"""
    try:
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return default or {}
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return default or {}

def save_prediction_to_history(text: str, prediction: str, confidence: float):
    """Save prediction to session history"""
    prediction_entry = {
        'timestamp': datetime.now().isoformat(),
        'text': text[:100] + "..." if len(text) > 100 else text,
        'prediction': prediction,
        'confidence': confidence,
        'text_length': len(text)
    }
    
    st.session_state.prediction_history.append(prediction_entry)
    
    # Keep only last 50 predictions
    if len(st.session_state.prediction_history) > 50:
        st.session_state.prediction_history = st.session_state.prediction_history[-50:]

def make_prediction_request(text: str) -> Dict[str, Any]:
    """Make prediction request to API"""
    try:
        if not app_manager.api_available:
            return {'error': 'API is not available'}
        
        response = app_manager.session.post(
            f"{app_manager.config['api_url']}/predict",
            json={"text": text},
            timeout=app_manager.config['prediction_timeout']
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {'error': f'API Error: {response.status_code} - {response.text}'}
            
    except requests.exceptions.Timeout:
        return {'error': 'Request timed out. Please try again.'}
    except requests.exceptions.ConnectionError:
        return {'error': 'Cannot connect to prediction service.'}
    except Exception as e:
        return {'error': f'Unexpected error: {str(e)}'}

def validate_text_input(text: str) -> tuple[bool, str]:
    """Validate text input"""
    if not text or not text.strip():
        return False, "Please enter some text to analyze."
    
    if len(text) < 10:
        return False, "Text must be at least 10 characters long."
    
    if len(text) > app_manager.config['max_text_length']:
        return False, f"Text must be less than {app_manager.config['max_text_length']} characters."
    
    # Check for suspicious content
    suspicious_patterns = ['<script', 'javascript:', 'data:']
    if any(pattern in text.lower() for pattern in suspicious_patterns):
        return False, "Text contains suspicious content."
    
    return True, "Valid"

def create_confidence_gauge(confidence: float, prediction: str):
    """Create confidence gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence: {prediction}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "red" if prediction == "Fake" else "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_prediction_history_chart():
    """Create prediction history visualization"""
    if not st.session_state.prediction_history:
        return None
    
    df = pd.DataFrame(st.session_state.prediction_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['confidence_percent'] = df['confidence'] * 100
    
    fig = px.scatter(
        df, 
        x='timestamp', 
        y='confidence_percent',
        color='prediction',
        size='text_length',
        hover_data=['text'],
        title="Prediction History",
        labels={'confidence_percent': 'Confidence (%)', 'timestamp': 'Time'}
    )
    
    fig.update_layout(height=400)
    return fig

# Main application
def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📰 Fake News Detection System</h1>', unsafe_allow_html=True)
    
    # API Status indicator
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if app_manager.api_available:
            st.markdown('<div class="success-message">🟢 API Service: Online</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-message">🔴 API Service: Offline</div>', unsafe_allow_html=True)
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Prediction", 
        "📊 Batch Analysis", 
        "📈 Analytics", 
        "🎯 Model Training", 
        "⚙️ System Status"
    ])
    
    # Tab 1: Individual Prediction
    with tab1:
        st.header("Single Text Analysis")
        
        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["Type Text", "Upload File"],
            horizontal=True
        )
        
        user_text = ""
        
        if input_method == "Type Text":
            user_text = st.text_area(
                "Enter news article text:",
                height=200,
                placeholder="Paste or type the news article you want to analyze..."
            )
        
        else:  # Upload File
            uploaded_file = st.file_uploader(
                "Upload text file:",
                type=['txt', 'csv'],
                help="Upload a text file containing the article to analyze"
            )
            
            if uploaded_file:
                try:
                    if uploaded_file.type == "text/plain":
                        user_text = str(uploaded_file.read(), "utf-8")
                    elif uploaded_file.type == "text/csv":
                        df = pd.read_csv(uploaded_file)
                        if 'text' in df.columns:
                            user_text = df['text'].iloc[0] if len(df) > 0 else ""
                        else:
                            st.error("CSV file must contain a 'text' column")
                    
                    st.success(f"File uploaded successfully! ({len(user_text)} characters)")
                
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        # Prediction section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🧠 Analyze Text", type="primary", use_container_width=True):
                if user_text:
                    # Validate input
                    is_valid, validation_message = validate_text_input(user_text)
                    
                    if not is_valid:
                        st.error(validation_message)
                    else:
                        # Show progress
                        with st.spinner("Analyzing text..."):
                            result = make_prediction_request(user_text)
                        
                        if 'error' in result:
                            st.error(f"❌ {result['error']}")
                        else:
                            # Display results
                            prediction = result['prediction']
                            confidence = result['confidence']
                            
                            # Save to history
                            save_prediction_to_history(user_text, prediction, confidence)
                            
                            # Results display
                            col_result1, col_result2 = st.columns(2)
                            
                            with col_result1:
                                if prediction == "Fake":
                                    st.markdown(f"""
                                    <div class="error-message">
                                        <h3>🚨 Prediction: FAKE NEWS</h3>
                                        <p>Confidence: {confidence:.2%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="success-message">
                                        <h3>✅ Prediction: REAL NEWS</h3>
                                        <p>Confidence: {confidence:.2%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            with col_result2:
                                # Confidence gauge
                                fig_gauge = create_confidence_gauge(confidence, prediction)
                                st.plotly_chart(fig_gauge, use_container_width=True)
                            
                            # Additional information
                            with st.expander("📋 Analysis Details"):
                                st.json({
                                    "model_version": result.get('model_version', 'Unknown'),
                                    "processing_time": f"{result.get('processing_time', 0):.3f} seconds",
                                    "timestamp": result.get('timestamp', ''),
                                    "text_length": len(user_text),
                                    "word_count": len(user_text.split())
                                })
                else:
                    st.warning("Please enter text to analyze.")
        
        with col2:
            if st.button("🔄 Clear Text", use_container_width=True):
                st.rerun()
    
    # Tab 2: Batch Analysis
    with tab2:
        st.header("Batch Text Analysis")
        
        # File upload for batch processing
        batch_file = st.file_uploader(
            "Upload CSV file for batch analysis:",
            type=['csv'],
            help="CSV file should contain a 'text' column with articles to analyze"
        )
        
        if batch_file:
            try:
                df = pd.read_csv(batch_file)
                
                if 'text' not in df.columns:
                    st.error("CSV file must contain a 'text' column")
                else:
                    st.success(f"File loaded: {len(df)} articles found")
                    
                    # Preview data
                    st.subheader("Data Preview")
                    st.dataframe(df.head(10))
                    
                    # Batch processing
                    if st.button("🚀 Process Batch", type="primary"):
                        if len(df) > app_manager.config['max_batch_size']:
                            st.warning(f"Only processing first {app_manager.config['max_batch_size']} articles")
                            df = df.head(app_manager.config['max_batch_size'])
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results = []
                        
                        for i, row in df.iterrows():
                            status_text.text(f"Processing article {i+1}/{len(df)}...")
                            progress_bar.progress((i + 1) / len(df))
                            
                            result = make_prediction_request(row['text'])
                            
                            if 'error' not in result:
                                results.append({
                                    'text': row['text'][:100] + "...",
                                    'prediction': result['prediction'],
                                    'confidence': result['confidence'],
                                    'processing_time': result.get('processing_time', 0)
                                })
                            else:
                                results.append({
                                    'text': row['text'][:100] + "...",
                                    'prediction': 'Error',
                                    'confidence': 0,
                                    'processing_time': 0
                                })
                        
                        # Display results
                        results_df = pd.DataFrame(results)
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Processed", len(results_df))
                        
                        with col2:
                            fake_count = len(results_df[results_df['prediction'] == 'Fake'])
                            st.metric("Fake News", fake_count)
                        
                        with col3:
                            real_count = len(results_df[results_df['prediction'] == 'Real'])
                            st.metric("Real News", real_count)
                        
                        with col4:
                            avg_confidence = results_df['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                        
                        # Results visualization
                        if len(results_df) > 0:
                            fig = px.histogram(
                                results_df,
                                x='prediction',
                                color='prediction',
                                title="Batch Analysis Results"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)
                        
                        st.download_button(
                            label="📥 Download Results",
                            data=csv_buffer.getvalue(),
                            file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # Tab 3: Analytics
    with tab3:
        st.header("System Analytics")
        
        # Prediction history
        if st.session_state.prediction_history:
            st.subheader("Recent Predictions")
            
            # History chart
            fig_history = create_prediction_history_chart()
            if fig_history:
                st.plotly_chart(fig_history, use_container_width=True)
            
            # History table
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df.tail(20), use_container_width=True)
        
        else:
            st.info("No prediction history available. Make some predictions to see analytics.")
        
        # System metrics
        st.subheader("System Metrics")
        
        # Load various log files for analytics
        try:
            # API health check
            if app_manager.api_available:
                response = app_manager.session.get(f"{app_manager.config['api_url']}/metrics")
                if response.status_code == 200:
                    metrics = response.json()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total API Requests", metrics.get('total_requests', 0))
                    
                    with col2:
                        st.metric("Unique Clients", metrics.get('unique_clients', 0))
                    
                    with col3:
                        st.metric("Model Version", metrics.get('model_version', 'Unknown'))
                    
                    with col4:
                        status = metrics.get('model_health', 'unknown')
                        st.metric("Model Status", status)
        
        except Exception as e:
            st.warning(f"Could not load API metrics: {e}")
    
    # Tab 4: Model Training
    with tab4:
        st.header("Custom Model Training")
        
        st.info("Upload your own dataset to retrain the model with custom data.")
        
        # File upload for training
        training_file = st.file_uploader(
            "Upload training dataset (CSV):",
            type=['csv'],
            help="CSV file should contain 'text' and 'label' columns (label: 0=Real, 1=Fake)"
        )
        
        if training_file:
            try:
                df_train = pd.read_csv(training_file)
                
                required_columns = ['text', 'label']
                missing_columns = [col for col in required_columns if col not in df_train.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {missing_columns}")
                else:
                    st.success(f"Training file loaded: {len(df_train)} samples")
                    
                    # Data validation
                    label_counts = df_train['label'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Dataset Overview")
                        st.write(f"Total samples: {len(df_train)}")
                        st.write(f"Real news (0): {label_counts.get(0, 0)}")
                        st.write(f"Fake news (1): {label_counts.get(1, 0)}")
                    
                    with col2:
                        # Label distribution chart
                        fig_labels = px.pie(
                            values=label_counts.values,
                            names=['Real', 'Fake'],
                            title="Label Distribution"
                        )
                        st.plotly_chart(fig_labels)
                    
                    # Training options
                    st.subheader("Training Configuration")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
                        max_features = st.number_input("Max Features", 1000, 20000, 10000, 1000)
                    
                    with col2:
                        cross_validation = st.checkbox("Cross Validation", value=True)
                        hyperparameter_tuning = st.checkbox("Hyperparameter Tuning", value=False)
                    
                    # Start training
                    if st.button("🏃‍♂️ Start Training", type="primary"):
                        # Save training data
                        app_manager.paths['custom_data'].parent.mkdir(parents=True, exist_ok=True)
                        df_train.to_csv(app_manager.paths['custom_data'], index=False)
                        
                        # Progress simulation
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        training_steps = [
                            "Preprocessing data...",
                            "Splitting dataset...",
                            "Training model...",
                            "Evaluating performance...",
                            "Saving model..."
                        ]
                        
                        for i, step in enumerate(training_steps):
                            status_text.text(step)
                            progress_bar.progress((i + 1) / len(training_steps))
                            time.sleep(2)  # Simulate processing time
                        
                        # Run actual training
                        try:
                            result = subprocess.run(
                                [sys.executable, "model/train.py", 
                                 "--data_path", str(app_manager.paths['custom_data'])],
                                capture_output=True,
                                text=True,
                                timeout=300
                            )
                            
                            if result.returncode == 0:
                                st.success("🎉 Training completed successfully!")
                                
                                # Try to extract accuracy from output
                                try:
                                    output_lines = result.stdout.strip().split('\n')
                                    for line in output_lines:
                                        if 'accuracy' in line.lower():
                                            st.info(f"Model performance: {line}")
                                except:
                                    pass
                                
                                # Reload API model
                                if app_manager.api_available:
                                    try:
                                        reload_response = app_manager.session.post(
                                            f"{app_manager.config['api_url']}/model/reload"
                                        )
                                        if reload_response.status_code == 200:
                                            st.success("✅ Model reloaded in API successfully!")
                                    except:
                                        st.warning("⚠️ Model trained but API reload failed")
                                
                            else:
                                st.error(f"Training failed: {result.stderr}")
                        
                        except subprocess.TimeoutExpired:
                            st.error("Training timed out. Please try with a smaller dataset.")
                        except Exception as e:
                            st.error(f"Training error: {e}")
            
            except Exception as e:
                st.error(f"Error loading training file: {e}")
    
    # Tab 5: System Status
    with tab5:
        render_system_status()

def render_system_status():
    """Render system status tab"""
    st.header("System Status & Monitoring")
    
    # Auto-refresh toggle
    col1, col2 = st.columns([1, 4])
    with col1:
        st.session_state.auto_refresh = st.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
    
    with col2:
        if st.button("🔄 Refresh Now"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    # System health overview
    st.subheader("🏥 System Health")
    
    if app_manager.api_available:
        try:
            health_response = app_manager.session.get(f"{app_manager.config['api_url']}/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                
                # Overall status
                overall_status = health_data.get('status', 'unknown')
                if overall_status == 'healthy':
                    st.success("🟢 System Status: Healthy")
                else:
                    st.error("🔴 System Status: Unhealthy")
                
                # Detailed health metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("🤖 Model Health")
                    model_health = health_data.get('model_health', {})
                    
                    for key, value in model_health.items():
                        if key != 'test_prediction':
                            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                with col2:
                    st.subheader("💻 System Resources")
                    system_health = health_data.get('system_health', {})
                    
                    for key, value in system_health.items():
                        if isinstance(value, (int, float)):
                            st.metric(key.replace('_', ' ').title(), f"{value:.1f}%")
                
                with col3:
                    st.subheader("🔗 API Health")
                    api_health = health_data.get('api_health', {})
                    
                    for key, value in api_health.items():
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        
        except Exception as e:
            st.error(f"Failed to get health status: {e}")
    
    else:
        st.error("🔴 API Service is not available")
    
    # Model information
    st.subheader("🎯 Model Information")
    
    metadata = load_json_file(app_manager.paths['metadata'], {})
    if metadata:
        col1, col2 = st.columns(2)
        
        with col1:
            for key in ['model_version', 'test_accuracy', 'test_f1', 'model_type']:
                if key in metadata:
                    display_key = key.replace('_', ' ').title()
                    value = metadata[key]
                    if isinstance(value, float):
                        st.metric(display_key, f"{value:.4f}")
                    else:
                        st.metric(display_key, str(value))
        
        with col2:
            for key in ['train_size', 'timestamp', 'data_version']:
                if key in metadata:
                    display_key = key.replace('_', ' ').title()
                    value = metadata[key]
                    if key == 'timestamp':
                        try:
                            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            value = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            pass
                    st.write(f"**{display_key}:** {value}")
    
    else:
        st.warning("No model metadata available")
    
    # Recent activity
    st.subheader("📜 Recent Activity")
    
    activity_log = load_json_file(app_manager.paths['activity_log'], [])
    if activity_log:
        recent_activities = activity_log[-10:] if len(activity_log) > 10 else activity_log
        
        for entry in reversed(recent_activities):
            timestamp = entry.get('timestamp', 'Unknown')
            event = entry.get('event', 'Unknown event')
            level = entry.get('level', 'INFO')
            
            if level == 'ERROR':
                st.error(f"🔴 {timestamp} - {event}")
            elif level == 'WARNING':
                st.warning(f"🟡 {timestamp} - {event}")
            else:
                st.info(f"🔵 {timestamp} - {event}")
    
    else:
        st.info("No recent activity logs found")
    
    # File system status
    st.subheader("📁 File System Status")
    
    critical_files = [
        ("/tmp/model.pkl", "Main Model"),
        ("/tmp/vectorizer.pkl", "Vectorizer"),
        ("/tmp/data/combined_dataset.csv", "Training Dataset"),
        ("/tmp/metadata.json", "Model Metadata")
    ]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Critical Files:**")
        for file_path, description in critical_files:
            if Path(file_path).exists():
                st.success(f"✅ {description}")
            else:
                st.error(f"❌ {description}")
    
    with col2:
        # Disk usage information
        try:
            import shutil
            total, used, free = shutil.disk_usage("/tmp")
            
            st.write("**Disk Usage (/tmp):**")
            st.write(f"Total: {total // (1024**3)} GB")
            st.write(f"Used: {used // (1024**3)} GB")
            st.write(f"Free: {free // (1024**3)} GB")
            
            usage_percent = (used / total) * 100
            if usage_percent > 90:
                st.error(f"⚠️ Disk usage: {usage_percent:.1f}%")
            elif usage_percent > 75:
                st.warning(f"⚠️ Disk usage: {usage_percent:.1f}%")
            else:
                st.success(f"✅ Disk usage: {usage_percent:.1f}%")
        
        except Exception as e:
            st.error(f"Cannot check disk usage: {e}")
    
    # Initialize system button
    if st.button("🔧 Initialize System", help="Run system initialization if components are missing"):
        with st.spinner("Running system initialization..."):
            try:
                result = subprocess.run(
                    [sys.executable, "/app/initialize_system.py"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    st.success("✅ System initialization completed successfully!")
                    st.code(result.stdout)
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("❌ System initialization failed")
                    st.code(result.stderr)
            
            except subprocess.TimeoutExpired:
                st.error("⏰ Initialization timed out")
            except Exception as e:
                st.error(f"❌ Initialization error: {e}")

# Auto-refresh logic
if st.session_state.auto_refresh:
    time_since_refresh = datetime.now() - st.session_state.last_refresh
    if time_since_refresh > timedelta(seconds=app_manager.config['refresh_interval']):
        st.session_state.last_refresh = datetime.now()
        st.rerun()

# Run main application
if __name__ == "__main__":
    main()