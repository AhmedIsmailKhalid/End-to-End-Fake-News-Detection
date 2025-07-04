import pandas as pd
import numpy as np
import joblib
import json
import logging
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectKBest, chi2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/model_retraining.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RobustModelRetrainer:
    """Production-ready model retraining with statistical validation and A/B testing"""
    
    def __init__(self):
        self.setup_paths()
        self.setup_retraining_config()
        self.setup_statistical_tests()
    
    def setup_paths(self):
        """Setup all necessary paths"""
        self.base_dir = Path("/tmp")
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "model"
        self.logs_dir = self.base_dir / "logs"
        self.backup_dir = self.base_dir / "backups"
        
        # Create directories
        for dir_path in [self.data_dir, self.model_dir, self.logs_dir, self.backup_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Current production files
        self.prod_model_path = self.model_dir / "model.pkl"
        self.prod_vectorizer_path = self.model_dir / "vectorizer.pkl"
        self.prod_pipeline_path = self.model_dir / "pipeline.pkl"
        
        # Candidate files
        self.candidate_model_path = self.model_dir / "model_candidate.pkl"
        self.candidate_vectorizer_path = self.model_dir / "vectorizer_candidate.pkl"
        self.candidate_pipeline_path = self.model_dir / "pipeline_candidate.pkl"
        
        # Data files
        self.combined_data_path = self.data_dir / "combined_dataset.csv"
        self.scraped_data_path = self.data_dir / "scraped_real.csv"
        self.generated_data_path = self.data_dir / "generated_fake.csv"
        
        # Metadata and logs
        self.metadata_path = Path("/tmp/metadata.json")
        self.retraining_log_path = self.logs_dir / "retraining_log.json"
        self.comparison_log_path = self.logs_dir / "model_comparison.json"
    
    def setup_retraining_config(self):
        """Setup retraining configuration"""
        self.min_new_samples = 50
        self.improvement_threshold = 0.01  # 1% improvement required
        self.significance_level = 0.05
        self.cv_folds = 5
        self.test_size = 0.2
        self.random_state = 42
        self.max_retries = 3
        self.backup_retention_days = 30
    
    def setup_statistical_tests(self):
        """Setup statistical test configurations"""
        self.statistical_tests = {
            'mcnemar': {'alpha': 0.05, 'name': "McNemar's Test"},
            'paired_ttest': {'alpha': 0.05, 'name': "Paired T-Test"},
            'wilcoxon': {'alpha': 0.05, 'name': "Wilcoxon Signed-Rank Test"}
        }
    
    def load_existing_metadata(self) -> Optional[Dict]:
        """Load existing model metadata"""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded existing metadata: {metadata.get('model_version', 'Unknown')}")
                return metadata
            else:
                logger.warning("No existing metadata found")
                return None
        except Exception as e:
            logger.error(f"Failed to load metadata: {str(e)}")
            return None
    
    def load_production_model(self) -> Tuple[bool, Optional[Any], str]:
        """Load current production model"""
        try:
            # Try to load pipeline first (preferred)
            if self.prod_pipeline_path.exists():
                model = joblib.load(self.prod_pipeline_path)
                logger.info("Loaded production pipeline")
                return True, model, "Pipeline loaded successfully"
            
            # Fallback to individual components
            elif self.prod_model_path.exists() and self.prod_vectorizer_path.exists():
                model = joblib.load(self.prod_model_path)
                vectorizer = joblib.load(self.prod_vectorizer_path)
                logger.info("Loaded production model and vectorizer")
                return True, (model, vectorizer), "Model components loaded successfully"
            
            else:
                return False, None, "No production model found"
                
        except Exception as e:
            error_msg = f"Failed to load production model: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def load_new_data(self) -> Tuple[bool, Optional[pd.DataFrame], str]:
        """Load and combine all available data"""
        try:
            logger.info("Loading training data...")
            
            dataframes = []
            
            # Load combined dataset (base)
            if self.combined_data_path.exists():
                df_combined = pd.read_csv(self.combined_data_path)
                dataframes.append(df_combined)
                logger.info(f"Loaded combined dataset: {len(df_combined)} samples")
            
            # Load scraped real news
            if self.scraped_data_path.exists():
                df_scraped = pd.read_csv(self.scraped_data_path)
                if 'label' not in df_scraped.columns:
                    df_scraped['label'] = 0  # Real news
                dataframes.append(df_scraped)
                logger.info(f"Loaded scraped data: {len(df_scraped)} samples")
            
            # Load generated fake news
            if self.generated_data_path.exists():
                df_generated = pd.read_csv(self.generated_data_path)
                if 'label' not in df_generated.columns:
                    df_generated['label'] = 1  # Fake news
                dataframes.append(df_generated)
                logger.info(f"Loaded generated data: {len(df_generated)} samples")
            
            if not dataframes:
                return False, None, "No data files found"
            
            # Combine all data
            df = pd.concat(dataframes, ignore_index=True)
            
            # Data cleaning and validation
            df = self.clean_and_validate_data(df)
            
            if len(df) < 100:
                return False, None, f"Insufficient data after cleaning: {len(df)} samples"
            
            logger.info(f"Total training data: {len(df)} samples")
            return True, df, f"Successfully loaded {len(df)} samples"
            
        except Exception as e:
            error_msg = f"Failed to load data: {str(e)}"
            logger.error(error_msg)
            return False, None, error_msg
    
    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the training data"""
        initial_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['text'], keep='first')
        
        # Remove null values
        df = df.dropna(subset=['text', 'label'])
        
        # Validate text quality
        df = df[df['text'].astype(str).str.len() > 10]
        
        # Validate labels
        df = df[df['label'].isin([0, 1])]
        
        # Remove excessive length texts
        df = df[df['text'].astype(str).str.len() < 10000]
        
        logger.info(f"Data cleaning: {initial_count} -> {len(df)} samples")
        return df
    
    def create_advanced_pipeline(self) -> Pipeline:
        """Create advanced ML pipeline"""
        def preprocess_text(texts):
            import re
            processed = []
            for text in texts:
                text = str(text)
                # Remove URLs and email addresses
                text = re.sub(r'http\S+|www\S+|https\S+|\S+@\S+', '', text)
                # Remove excessive punctuation
                text = re.sub(r'[!]{2,}', '!', text)
                text = re.sub(r'[?]{2,}', '?', text)
                # Remove non-alphabetic characters except spaces and punctuation
                text = re.sub(r'[^a-zA-Z\s.!?]', '', text)
                # Remove excessive whitespace
                text = re.sub(r'\s+', ' ', text)
                processed.append(text.strip().lower())
            return processed
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocess', FunctionTransformer(preprocess_text, validate=False)),
            ('vectorize', TfidfVectorizer(
                max_features=10000,
                min_df=2,
                max_df=0.95,
                ngram_range=(1, 3),
                stop_words='english',
                sublinear_tf=True
            )),
            ('feature_select', SelectKBest(chi2, k=5000)),
            ('model', LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state
            ))
        ])
        
        return pipeline
    
    def train_candidate_model(self, df: pd.DataFrame) -> Tuple[bool, Optional[Any], Dict]:
        """Train candidate model with comprehensive evaluation"""
        try:
            logger.info("Training candidate model...")
            
            # Prepare data
            X = df['text'].values
            y = df['label'].values
            
            # Train-test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
            )
            
            # Create and train pipeline
            pipeline = self.create_advanced_pipeline()
            pipeline.fit(X_train, y_train)
            
            # Evaluate candidate model
            evaluation_results = self.evaluate_model(pipeline, X_test, y_test, X_train, y_train)
            
            # Save candidate model
            joblib.dump(pipeline, self.candidate_pipeline_path)
            joblib.dump(pipeline.named_steps['model'], self.candidate_model_path)
            joblib.dump(pipeline.named_steps['vectorize'], self.candidate_vectorizer_path)
            
            logger.info(f"Candidate model training completed")
            logger.info(f"Candidate F1 Score: {evaluation_results['f1']:.4f}")
            logger.info(f"Candidate Accuracy: {evaluation_results['accuracy']:.4f}")
            
            return True, pipeline, evaluation_results
            
        except Exception as e:
            error_msg = f"Candidate model training failed: {str(e)}"
            logger.error(error_msg)
            return False, None, {'error': error_msg}
    
    def evaluate_model(self, model, X_test, y_test, X_train=None, y_train=None) -> Dict:
        """Comprehensive model evaluation"""
        try:
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Basic metrics
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted')),
                'recall': float(recall_score(y_test, y_pred, average='weighted')),
                'f1': float(f1_score(y_test, y_pred, average='weighted')),
                'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            # Cross-validation
            if X_train is not None and y_train is not None:
                try:
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
                        scoring='f1_weighted'
                    )
                    metrics['cv_f1_mean'] = float(cv_scores.mean())
                    metrics['cv_f1_std'] = float(cv_scores.std())
                except Exception as e:
                    logger.warning(f"Cross-validation failed: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return {'error': str(e)}
    
    def compare_models_statistically(self, prod_model, candidate_model, X_test, y_test) -> Dict:
        """Statistical comparison of models"""
        try:
            logger.info("Performing statistical model comparison...")
            
            # Get predictions
            prod_pred = prod_model.predict(X_test)
            candidate_pred = candidate_model.predict(X_test)
            
            # Calculate accuracies
            prod_accuracy = accuracy_score(y_test, prod_pred)
            candidate_accuracy = accuracy_score(y_test, candidate_pred)
            
            comparison_results = {
                'production_accuracy': float(prod_accuracy),
                'candidate_accuracy': float(candidate_accuracy),
                'absolute_improvement': float(candidate_accuracy - prod_accuracy),
                'relative_improvement': float((candidate_accuracy - prod_accuracy) / prod_accuracy * 100),
                'statistical_tests': {}
            }
            
            # McNemar's test for paired predictions
            try:
                # Create contingency table
                prod_correct = (prod_pred == y_test)
                candidate_correct = (candidate_pred == y_test)
                
                both_correct = np.sum(prod_correct & candidate_correct)
                prod_only = np.sum(prod_correct & ~candidate_correct)
                candidate_only = np.sum(~prod_correct & candidate_correct)
                both_wrong = np.sum(~prod_correct & ~candidate_correct)
                
                # McNemar's test
                if prod_only + candidate_only > 0:
                    mcnemar_stat = (abs(prod_only - candidate_only) - 1) ** 2 / (prod_only + candidate_only)
                    p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
                    
                    comparison_results['statistical_tests']['mcnemar'] = {
                        'statistic': float(mcnemar_stat),
                        'p_value': float(p_value),
                        'significant': p_value < self.significance_level,
                        'contingency_table': {
                            'both_correct': int(both_correct),
                            'prod_only': int(prod_only),
                            'candidate_only': int(candidate_only),
                            'both_wrong': int(both_wrong)
                        }
                    }
                
            except Exception as e:
                logger.warning(f"McNemar's test failed: {e}")
            
            # Practical significance test
            comparison_results['practical_significance'] = {
                'meets_threshold': comparison_results['absolute_improvement'] >= self.improvement_threshold,
                'threshold': self.improvement_threshold,
                'recommendation': 'promote' if (
                    comparison_results['absolute_improvement'] >= self.improvement_threshold and
                    comparison_results['statistical_tests'].get('mcnemar', {}).get('significant', False)
                ) else 'keep_current'
            }
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Statistical comparison failed: {str(e)}")
            return {'error': str(e)}
    
    def create_backup(self) -> bool:
        """Create backup of current production model"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = self.backup_dir / f"backup_{timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup files
            files_to_backup = [
                (self.prod_model_path, backup_dir / "model.pkl"),
                (self.prod_vectorizer_path, backup_dir / "vectorizer.pkl"),
                (self.prod_pipeline_path, backup_dir / "pipeline.pkl"),
                (self.metadata_path, backup_dir / "metadata.json")
            ]
            
            for source, dest in files_to_backup:
                if source.exists():
                    shutil.copy2(source, dest)
            
            logger.info(f"Backup created: {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Backup creation failed: {str(e)}")
            return False
    
    def promote_candidate_model(self, candidate_model, candidate_metrics: Dict, comparison_results: Dict) -> bool:
        """Promote candidate model to production"""
        try:
            logger.info("Promoting candidate model to production...")
            
            # Create backup first
            if not self.create_backup():
                logger.error("Backup creation failed, aborting promotion")
                return False
            
            # Copy candidate files to production
            shutil.copy2(self.candidate_model_path, self.prod_model_path)
            shutil.copy2(self.candidate_vectorizer_path, self.prod_vectorizer_path)
            shutil.copy2(self.candidate_pipeline_path, self.prod_pipeline_path)
            
            # Update metadata
            metadata = self.load_existing_metadata() or {}
            
            # Increment version
            old_version = metadata.get('model_version', 'v1.0')
            if old_version.startswith('v'):
                try:
                    major, minor = map(int, old_version[1:].split('.'))
                    new_version = f"v{major}.{minor + 1}"
                except:
                    new_version = f"v1.{int(datetime.now().timestamp()) % 1000}"
            else:
                new_version = f"v1.{int(datetime.now().timestamp()) % 1000}"
            
            # Update metadata
            metadata.update({
                'model_version': new_version,
                'model_type': 'retrained_pipeline',
                'previous_version': old_version,
                'test_accuracy': candidate_metrics['accuracy'],
                'test_f1': candidate_metrics['f1'],
                'test_precision': candidate_metrics['precision'],
                'test_recall': candidate_metrics['recall'],
                'test_roc_auc': candidate_metrics['roc_auc'],
                'improvement_over_previous': comparison_results['absolute_improvement'],
                'statistical_significance': comparison_results['statistical_tests'].get('mcnemar', {}).get('significant', False),
                'promotion_timestamp': datetime.now().isoformat(),
                'retrain_trigger': 'scheduled_retrain'
            })
            
            # Save updated metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model promoted successfully to {new_version}")
            return True
            
        except Exception as e:
            logger.error(f"Model promotion failed: {str(e)}")
            return False
    
    def log_retraining_session(self, results: Dict):
        """Log retraining session results"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'session_id': hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
            }
            
            # Load existing logs
            logs = []
            if self.retraining_log_path.exists():
                try:
                    with open(self.retraining_log_path, 'r') as f:
                        logs = json.load(f)
                except:
                    logs = []
            
            # Add new log
            logs.append(log_entry)
            
            # Keep only last 100 entries
            if len(logs) > 100:
                logs = logs[-100:]
            
            # Save logs
            with open(self.retraining_log_path, 'w') as f:
                json.dump(logs, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to log retraining session: {str(e)}")
    
    def retrain_model(self) -> Tuple[bool, str]:
        """Main retraining function with comprehensive validation"""
        try:
            logger.info("Starting model retraining process...")
            
            # Load existing metadata
            existing_metadata = self.load_existing_metadata()
            
            # Load production model
            prod_success, prod_model, prod_msg = self.load_production_model()
            if not prod_success:
                logger.warning(f"No production model found: {prod_msg}")
                # Fall back to initial training
                from model.train import main as train_main
                train_main()
                return True, "Initial training completed"
            
            # Load new data
            data_success, df, data_msg = self.load_new_data()
            if not data_success:
                return False, data_msg
            
            # Check if we have enough new data
            if len(df) < self.min_new_samples:
                return False, f"Insufficient new data: {len(df)} < {self.min_new_samples}"
            
            # Train candidate model
            candidate_success, candidate_model, candidate_metrics = self.train_candidate_model(df)
            if not candidate_success:
                return False, f"Candidate training failed: {candidate_metrics.get('error', 'Unknown error')}"
            
            # Prepare test data for comparison
            X = df['text'].values
            y = df['label'].values
            from sklearn.model_selection import train_test_split
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
            )
            
            # Compare models
            comparison_results = self.compare_models_statistically(
                prod_model, candidate_model, X_test, y_test
            )
            
            # Log results
            session_results = {
                'candidate_metrics': candidate_metrics,
                'comparison_results': comparison_results,
                'data_size': len(df),
                'test_size': len(X_test)
            }
            
            self.log_retraining_session(session_results)
            
            # Decide whether to promote
            should_promote = (
                comparison_results['absolute_improvement'] >= self.improvement_threshold and
                comparison_results.get('statistical_tests', {}).get('mcnemar', {}).get('significant', False)
            )
            
            if should_promote:
                # Promote candidate model
                promotion_success = self.promote_candidate_model(
                    candidate_model, candidate_metrics, comparison_results
                )
                
                if promotion_success:
                    success_msg = (
                        f"Model promoted successfully! "
                        f"Improvement: {comparison_results['absolute_improvement']:.4f} "
                        f"(F1: {candidate_metrics['f1']:.4f})"
                    )
                    logger.info(success_msg)
                    return True, success_msg
                else:
                    return False, "Model promotion failed"
            else:
                # Keep current model
                keep_msg = (
                    f"Keeping current model. "
                    f"Improvement: {comparison_results['absolute_improvement']:.4f} "
                    f"(threshold: {self.improvement_threshold})"
                )
                logger.info(keep_msg)
                return True, keep_msg
            
        except Exception as e:
            error_msg = f"Model retraining failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

def main():
    """Main execution function"""
    retrainer = RobustModelRetrainer()
    success, message = retrainer.retrain_model()
    
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
        exit(1)

if __name__ == "__main__":
    main()