import os
import sys
import shutil
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def log_step(message):
    """Log initialization steps"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def create_directories():
    """Create necessary directories"""
    log_step("Creating directory structure...")
    
    directories = [
        "/tmp/data",
        "/tmp/model", 
        "/tmp/logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        log_step(f"✅ Created {dir_path}")

def copy_original_datasets():
    """Copy original datasets from /app to /tmp"""
    log_step("Copying original datasets...")
    
    source_files = [
        ("/app/data/kaggle/Fake.csv", "/tmp/data/kaggle/Fake.csv"),
        ("/app/data/kaggle/True.csv", "/tmp/data/kaggle/True.csv"),
        ("/app/data/combined_dataset.csv", "/tmp/data/combined_dataset.csv")
    ]
    
    copied_count = 0
    for source, dest in source_files:
        if Path(source).exists():
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, dest)
            log_step(f"✅ Copied {source} to {dest}")
            copied_count += 1
        else:
            log_step(f"⚠️ Source file not found: {source}")
    
    return copied_count > 0

def create_minimal_dataset():
    """Create a minimal dataset if original doesn't exist"""
    log_step("Creating minimal dataset...")
    
    combined_path = Path("/tmp/data/combined_dataset.csv")
    
    if combined_path.exists():
        log_step("✅ Combined dataset already exists")
        return True
    
    # Create minimal training data
    minimal_data = pd.DataFrame({
        'text': [
            'Scientists discover new species in Amazon rainforest',
            'SHOCKING: Aliens spotted in Area 51, government confirms existence',
            'Local authorities report increase in renewable energy adoption',
            'You won\'t believe what happens when you eat this miracle fruit',
            'Economic indicators show steady growth in manufacturing sector',
            'EXCLUSIVE: Celebrity caught in secret alien communication scandal',
            'Research shows positive effects of meditation on mental health',
            'Government hiding truth about flat earth, conspiracy theorists claim',
            'New study reveals benefits of regular exercise for elderly',
            'BREAKING: Time travel confirmed by underground scientists'
        ],
        'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 0=Real, 1=Fake
    })
    
    minimal_data.to_csv(combined_path, index=False)
    log_step(f"✅ Created minimal dataset with {len(minimal_data)} samples")
    return True

def run_initial_training():
    """Run basic model training"""
    log_step("Starting initial model training...")
    
    try:
        # Check if model already exists
        model_path = Path("/tmp/model.pkl")
        vectorizer_path = Path("/tmp/vectorizer.pkl")
        
        if model_path.exists() and vectorizer_path.exists():
            log_step("✅ Model files already exist")
            return True
        
        # Import required libraries
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import joblib
        
        # Load dataset
        dataset_path = Path("/tmp/data/combined_dataset.csv")
        if not dataset_path.exists():
            log_step("❌ No dataset available for training")
            return False
        
        df = pd.read_csv(dataset_path)
        log_step(f"Loaded dataset with {len(df)} samples")
        
        # Prepare data
        X = df['text'].values
        y = df['label'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Vectorization
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        joblib.dump(model, "/tmp/model.pkl")
        joblib.dump(vectorizer, "/tmp/vectorizer.pkl")
        
        # Save metadata
        metadata = {
            "model_version": "v1.0_init",
            "test_accuracy": float(accuracy),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "timestamp": datetime.now().isoformat(),
            "training_method": "initialization"
        }
        
        with open("/tmp/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log_step(f"✅ Training completed successfully, accuracy: {accuracy:.4f}")
        return True
        
    except Exception as e:
        log_step(f"❌ Training failed: {str(e)}")
        return False

def create_initial_logs():
    """Create initial log files"""
    log_step("Creating initial log files...")
    
    try:
        # Activity log
        activity_log = [{
            "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
            "event": "System initialized successfully"
        }]
        
        with open("/tmp/activity_log.json", 'w') as f:
            json.dump(activity_log, f, indent=2)
        
        # Create empty monitoring logs
        with open("/tmp/logs/monitoring_log.json", 'w') as f:
            json.dump([], f)
        
        log_step("✅ Initial log files created")
        return True
        
    except Exception as e:
        log_step(f"❌ Log creation failed: {str(e)}")
        return False

def main():
    """Main initialization function"""
    log_step("🚀 Starting system initialization...")
    
    steps = [
        ("Directory Creation", create_directories),
        ("Dataset Copy", copy_original_datasets),
        ("Minimal Dataset", create_minimal_dataset),
        ("Model Training", run_initial_training),
        ("Log Creation", create_initial_logs)
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        try:
            if step_function():
                log_step(f"✅ {step_name} completed")
            else:
                log_step(f"❌ {step_name} failed")
                failed_steps.append(step_name)
        except Exception as e:
            log_step(f"❌ {step_name} failed: {str(e)}")
            failed_steps.append(step_name)
    
    if failed_steps:
        log_step(f"⚠️ Initialization completed with {len(failed_steps)} failed steps")
        log_step(f"Failed: {', '.join(failed_steps)}")
    else:
        log_step("🎉 System initialization completed successfully!")
    
    log_step("System ready for use!")

if __name__ == "__main__":
    main()