import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import joblib
import hashlib
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, validation_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RobustModelTrainer:
    """Production-ready model trainer with comprehensive evaluation and validation"""
    
    def __init__(self):
        self.setup_paths()
        self.setup_training_config()
        self.setup_models()
    
    def setup_paths(self):
        """Setup all necessary paths"""
        self.base_dir = Path("/tmp")
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "model"
        self.results_dir = self.base_dir / "results"
        
        # Create directories
        for dir_path in [self.data_dir, self.model_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.data_path = self.data_dir / "combined_dataset.csv"
        self.model_path = self.model_dir / "model.pkl"
        self.vectorizer_path = self.model_dir / "vectorizer.pkl"
        self.pipeline_path = self.model_dir / "pipeline.pkl"
        self.metadata_path = Path("/tmp/metadata.json")
        self.evaluation_path = self.results_dir / "evaluation_results.json"
    
    def setup_training_config(self):
        """Setup training configuration"""
        self.test_size = 0.2
        self.validation_size = 0.1
        self.random_state = 42
        self.cv_folds = 5
        self.max_features = 10000
        self.min_df = 2
        self.max_df = 0.95
        self.ngram_range = (1, 3)
        self.max_iter = 1000
        self.class_weight = 'balanced'
        self.feature_selection_k = 5000
    
    def setup_models(self):
        """Setup model configurations for comparison"""
        self.models = {
            'logistic_regression': {
                'model': LogisticRegression(
                    max_iter=self.max_iter,
                    class_weight=self.class_weight,
                    random_state=self.random_state
                ),
                'param_grid': {
                    'model__C': [0.1, 1, 10, 100],
                    'model__penalty': ['l2'],
                    'model__solver': ['liblinear', 'lbfgs']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    class_weight=self.class_weight,
                    random_state=self.random_state
                ),
                'param_grid': {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [10, 20, None],
                    'model__min_samples_split': [2, 5, 10]
                }
            }
        }
    
    def load_and_validate_data(self) -> Tuple[bool, Optional[pd.DataFrame], str]:
        """Load and validate training data"""
        try:
            logger.info("Loading training data...")
            
            if not self.data_path.exists():
                return False, None, f"Data file not found: {self.data_path}"
            
            # Load data
            df = pd.read_csv(self.data_path)
            
            # Basic validation
            if df.empty:
                return False, None, "Dataset is empty"
            
            required_columns = ['text', 'label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, None, f"Missing required columns: {missing_columns}"
            
            # Remove missing values
            initial_count = len(df)
            df = df.dropna(subset=required_columns)
            if len(df) < initial_count:
                logger.warning(f"Removed {initial_count - len(df)} rows with missing values")
            
            # Validate text content
            df = df[df['text'].astype(str).str.len() > 10]
            
            # Validate labels
            unique_labels = df['label'].unique()
            if len(unique_labels) < 2:
                return False, None, f"Need at least 2 classes, found: {unique_labels}"
            
            # Check minimum sample size
            if len(df) < 100:
                return False, None, f"Insufficient samples for training: {len(df)}"
            
            # Check class balance
            label_counts = df['label'].value_counts()
            min_class_ratio = label_counts.min() / label_counts.max()
            if min_class_ratio < 0.1:
                logger.warning(f"Severe class imbalance detected: {min_class_ratio:.3f}")
            
            logger.info(f"Data validation successful: {len(df)} samples, {len(unique_labels)} classes")
            logger.info(f"Class distribution: {label_counts.to_dict()}")
            
            return True, df, "Data loaded successfully"
            
        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        import re
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Remove non-alphabetic characters except spaces and basic punctuation
        text = re.sub(r'[^a-zA-Z\s.!?]', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip().lower()
    
    def create_preprocessing_pipeline(self) -> Pipeline:
        """Create advanced preprocessing pipeline"""
        # Text preprocessing
        text_preprocessor = FunctionTransformer(
            func=lambda x: [self.preprocess_text(text) for text in x],
            validate=False
        )
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            stop_words='english',
            sublinear_tf=True,
            norm='l2'
        )
        
        # Feature selection
        feature_selector = SelectKBest(
            score_func=chi2,
            k=self.feature_selection_k
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocess', text_preprocessor),
            ('vectorize', vectorizer),
            ('feature_select', feature_selector),
            ('model', None)  # Will be set during training
        ])
        
        return pipeline
    
    def comprehensive_evaluation(self, model, X_test, y_test, X_train=None, y_train=None) -> Dict:
        """Comprehensive model evaluation with multiple metrics"""
        logger.info("Starting comprehensive model evaluation...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Basic metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted')),
            'f1': float(f1_score(y_test, y_pred, average='weighted')),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = class_report
        
        # Cross-validation scores if training data provided
        if X_train is not None and y_train is not None:
            try:
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                    scoring='f1_weighted'
                )
                metrics['cv_scores'] = {
                    'mean': float(cv_scores.mean()),
                    'std': float(cv_scores.std()),
                    'scores': cv_scores.tolist()
                }
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")
                metrics['cv_scores'] = None
        
        # Feature importance (if available)
        try:
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                metrics['feature_importance_stats'] = {
                    'mean': float(feature_importance.mean()),
                    'std': float(feature_importance.std()),
                    'top_features': feature_importance.argsort()[-10:][::-1].tolist()
                }
            elif hasattr(model, 'coef_'):
                coefficients = model.coef_[0]
                metrics['coefficient_stats'] = {
                    'mean': float(coefficients.mean()),
                    'std': float(coefficients.std()),
                    'top_positive': coefficients.argsort()[-10:][::-1].tolist(),
                    'top_negative': coefficients.argsort()[:10].tolist()
                }
        except Exception as e:
            logger.warning(f"Feature importance extraction failed: {e}")
        
        # Model complexity metrics
        try:
            # Training accuracy for overfitting detection
            if X_train is not None and y_train is not None:
                y_train_pred = model.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                metrics['train_accuracy'] = float(train_accuracy)
                metrics['overfitting_score'] = float(train_accuracy - metrics['accuracy'])
        except Exception as e:
            logger.warning(f"Overfitting detection failed: {e}")
        
        return metrics
    
    def hyperparameter_tuning(self, pipeline, X_train, y_train, model_name: str) -> Tuple[Any, Dict]:
        """Perform hyperparameter tuning with cross-validation"""
        logger.info(f"Starting hyperparameter tuning for {model_name}...")
        
        try:
            # Set the model in the pipeline
            pipeline.set_params(model=self.models[model_name]['model'])
            
            # Get parameter grid
            param_grid = self.models[model_name]['param_grid']
            
            # Create GridSearchCV
            grid_search = GridSearchCV(
                pipeline,
                param_grid,
                cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            
            # Fit grid search
            grid_search.fit(X_train, y_train)
            
            # Extract results
            tuning_results = {
                'best_params': grid_search.best_params_,
                'best_score': float(grid_search.best_score_),
                'best_estimator': grid_search.best_estimator_,
                'cv_results': {
                    'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                    'std_test_scores': grid_search.cv_results_['std_test_score'].tolist(),
                    'params': grid_search.cv_results_['params']
                }
            }
            
            logger.info(f"Hyperparameter tuning completed for {model_name}")
            logger.info(f"Best score: {grid_search.best_score_:.4f}")
            logger.info(f"Best params: {grid_search.best_params_}")
            
            return grid_search.best_estimator_, tuning_results
            
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed for {model_name}: {str(e)}")
            # Return basic model if tuning fails
            pipeline.set_params(model=self.models[model_name]['model'])
            pipeline.fit(X_train, y_train)
            return pipeline, {'error': str(e)}
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test) -> Dict:
        """Train and evaluate multiple models"""
        logger.info("Starting model training and evaluation...")
        
        results = {}
        
        for model_name in self.models.keys():
            logger.info(f"Training {model_name}...")
            
            try:
                # Create pipeline
                pipeline = self.create_preprocessing_pipeline()
                
                # Hyperparameter tuning
                best_model, tuning_results = self.hyperparameter_tuning(
                    pipeline, X_train, y_train, model_name
                )
                
                # Comprehensive evaluation
                evaluation_metrics = self.comprehensive_evaluation(
                    best_model, X_test, y_test, X_train, y_train
                )
                
                # Store results
                results[model_name] = {
                    'model': best_model,
                    'tuning_results': tuning_results,
                    'evaluation_metrics': evaluation_metrics,
                    'training_time': datetime.now().isoformat()
                }
                
                logger.info(f"Model {model_name} - F1: {evaluation_metrics['f1']:.4f}, "
                           f"Accuracy: {evaluation_metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Training failed for {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def select_best_model(self, results: Dict) -> Tuple[str, Any, Dict]:
        """Select the best performing model"""
        logger.info("Selecting best model...")
        
        best_model_name = None
        best_model = None
        best_score = -1
        best_metrics = None
        
        for model_name, result in results.items():
            if 'error' in result:
                continue
            
            # Use F1 score as primary metric
            f1_score = result['evaluation_metrics']['f1']
            
            if f1_score > best_score:
                best_score = f1_score
                best_model_name = model_name
                best_model = result['model']
                best_metrics = result['evaluation_metrics']
        
        if best_model_name is None:
            raise ValueError("No models trained successfully")
        
        logger.info(f"Best model: {best_model_name} with F1 score: {best_score:.4f}")
        return best_model_name, best_model, best_metrics
    
    def save_model_artifacts(self, model, model_name: str, metrics: Dict) -> bool:
        """Save model artifacts and metadata"""
        try:
            logger.info("Saving model artifacts...")
            
            # Save the full pipeline
            joblib.dump(model, self.pipeline_path)
            
            # Save individual components for backward compatibility
            joblib.dump(model.named_steps['model'], self.model_path)
            joblib.dump(model.named_steps['vectorize'], self.vectorizer_path)
            
            # Generate data hash
            data_hash = hashlib.md5(str(datetime.now()).encode()).hexdigest()
            
            # Create metadata
            metadata = {
                'model_version': f"v1.0_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'model_type': model_name,
                'data_version': data_hash,
                'train_size': metrics.get('train_accuracy', 'Unknown'),
                'test_size': len(metrics.get('confusion_matrix', [[0]])[0]) if 'confusion_matrix' in metrics else 'Unknown',
                'test_accuracy': metrics['accuracy'],
                'test_f1': metrics['f1'],
                'test_precision': metrics['precision'],
                'test_recall': metrics['recall'],
                'test_roc_auc': metrics['roc_auc'],
                'overfitting_score': metrics.get('overfitting_score', 'Unknown'),
                'cv_score_mean': metrics.get('cv_scores', {}).get('mean', 'Unknown'),
                'cv_score_std': metrics.get('cv_scores', {}).get('std', 'Unknown'),
                'timestamp': datetime.now().isoformat(),
                'training_config': {
                    'test_size': self.test_size,
                    'validation_size': self.validation_size,
                    'cv_folds': self.cv_folds,
                    'max_features': self.max_features,
                    'ngram_range': self.ngram_range,
                    'feature_selection_k': self.feature_selection_k
                }
            }
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model artifacts saved successfully")
            logger.info(f"Model path: {self.model_path}")
            logger.info(f"Vectorizer path: {self.vectorizer_path}")
            logger.info(f"Pipeline path: {self.pipeline_path}")
            logger.info(f"Metadata path: {self.metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model artifacts: {str(e)}")
            return False
    
    def save_evaluation_results(self, results: Dict) -> bool:
        """Save comprehensive evaluation results"""
        try:
            # Clean results for JSON serialization
            clean_results = {}
            for model_name, result in results.items():
                if 'error' in result:
                    clean_results[model_name] = result
                else:
                    clean_results[model_name] = {
                        'tuning_results': {
                            k: v for k, v in result['tuning_results'].items() 
                            if k != 'best_estimator'
                        },
                        'evaluation_metrics': result['evaluation_metrics'],
                        'training_time': result['training_time']
                    }
            
            # Save results
            with open(self.evaluation_path, 'w') as f:
                json.dump(clean_results, f, indent=2, default=str)
            
            logger.info(f"Evaluation results saved to {self.evaluation_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {str(e)}")
            return False
    
    def train_model(self, data_path: str = None) -> Tuple[bool, str]:
        """Main training function with comprehensive pipeline"""
        try:
            logger.info("Starting model training pipeline...")
            
            # Override data path if provided
            if data_path:
                self.data_path = Path(data_path)
            
            # Load and validate data
            success, df, message = self.load_and_validate_data()
            if not success:
                return False, message
            
            # Prepare data
            X = df['text'].values
            y = df['label'].values
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size,
                stratify=y,
                random_state=self.random_state
            )
            
            logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test")
            
            # Train and evaluate models
            results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test)
            
            # Select best model
            best_model_name, best_model, best_metrics = self.select_best_model(results)
            
            # Save model artifacts
            if not self.save_model_artifacts(best_model, best_model_name, best_metrics):
                return False, "Failed to save model artifacts"
            
            # Save evaluation results
            self.save_evaluation_results(results)
            
            success_message = (
                f"Model training completed successfully. "
                f"Best model: {best_model_name} "
                f"(F1: {best_metrics['f1']:.4f}, Accuracy: {best_metrics['accuracy']:.4f})"
            )
            
            logger.info(success_message)
            return True, success_message
            
        except Exception as e:
            error_message = f"Model training failed: {str(e)}"
            logger.error(error_message)
            return False, error_message

def main():
    """Main execution function"""
    trainer = RobustModelTrainer()
    success, message = trainer.train_model()
    
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
        exit(1)

if __name__ == "__main__":
    main()