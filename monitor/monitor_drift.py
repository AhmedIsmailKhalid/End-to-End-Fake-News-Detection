import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

# Statistical imports
from scipy.spatial.distance import jensenshannon
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/drift_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedDriftMonitor:
    """Advanced drift detection with multiple statistical methods and comprehensive monitoring"""
    
    def __init__(self):
        self.setup_paths()
        self.setup_drift_config()
        self.setup_drift_methods()
        self.historical_data = self.load_historical_data()
    
    def setup_paths(self):
        """Setup all necessary paths"""
        self.base_dir = Path("/tmp")
        self.data_dir = self.base_dir / "data"
        self.model_dir = self.base_dir / "model"
        self.logs_dir = self.base_dir / "logs"
        self.results_dir = self.base_dir / "drift_results"
        
        # Create directories
        for dir_path in [self.data_dir, self.model_dir, self.logs_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Data files
        self.reference_data_path = self.data_dir / "combined_dataset.csv"
        self.current_data_path = self.data_dir / "scraped_real.csv"
        self.generated_data_path = self.data_dir / "generated_fake.csv"
        
        # Model files
        self.vectorizer_path = self.model_dir / "vectorizer.pkl"
        self.model_path = self.model_dir / "model.pkl"
        self.pipeline_path = self.model_dir / "pipeline.pkl"
        
        # Monitoring files
        self.drift_log_path = self.logs_dir / "monitoring_log.json"
        self.drift_history_path = self.logs_dir / "drift_history.json"
        self.alert_log_path = self.logs_dir / "drift_alerts.json"
    
    def setup_drift_config(self):
        """Setup drift detection configuration"""
        self.drift_thresholds = {
            'jensen_shannon': 0.1,
            'kolmogorov_smirnov': 0.05,
            'population_stability_index': 0.2,
            'performance_degradation': 0.05,
            'feature_drift': 0.1
        }
        
        self.alert_thresholds = {
            'high_drift': 0.3,
            'medium_drift': 0.15,
            'low_drift': 0.05
        }
        
        self.monitoring_config = {
            'min_samples': 100,
            'max_samples': 1000,
            'lookback_days': 30,
            'min_monitoring_interval': timedelta(hours=1),
            'confidence_level': 0.95
        }
    
    def setup_drift_methods(self):
        """Setup drift detection methods"""
        self.drift_methods = {
            'jensen_shannon': self.jensen_shannon_drift,
            'kolmogorov_smirnov': self.kolmogorov_smirnov_drift,
            'population_stability_index': self.population_stability_index_drift,
            'performance_drift': self.performance_drift,
            'feature_importance_drift': self.feature_importance_drift,
            'statistical_distance': self.statistical_distance_drift
        }
    
    def load_historical_data(self) -> Dict:
        """Load historical drift monitoring data"""
        try:
            if self.drift_history_path.exists():
                with open(self.drift_history_path, 'r') as f:
                    return json.load(f)
            return {'baseline_statistics': {}, 'historical_scores': []}
        except Exception as e:
            logger.warning(f"Failed to load historical data: {e}")
            return {'baseline_statistics': {}, 'historical_scores': []}
    
    def load_vectorizer(self) -> Optional[Any]:
        """Load the trained vectorizer"""
        try:
            # Try pipeline first
            if self.pipeline_path.exists():
                pipeline = joblib.load(self.pipeline_path)
                return pipeline.named_steps.get('vectorize') or pipeline.named_steps.get('vectorizer')
            
            # Fallback to individual vectorizer
            if self.vectorizer_path.exists():
                return joblib.load(self.vectorizer_path)
            
            logger.error("No vectorizer found")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load vectorizer: {e}")
            return None
    
    def load_model(self) -> Optional[Any]:
        """Load the trained model"""
        try:
            # Try pipeline first
            if self.pipeline_path.exists():
                return joblib.load(self.pipeline_path)
            
            # Fallback to individual model
            if self.model_path.exists():
                return joblib.load(self.model_path)
            
            logger.error("No model found")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def load_and_prepare_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load and prepare reference and current data"""
        try:
            # Load reference data
            reference_df = None
            if self.reference_data_path.exists():
                reference_df = pd.read_csv(self.reference_data_path)
                logger.info(f"Loaded reference data: {len(reference_df)} samples")
            
            # Load current data
            current_dfs = []
            
            if self.current_data_path.exists():
                df_current = pd.read_csv(self.current_data_path)
                current_dfs.append(df_current)
                logger.info(f"Loaded current scraped data: {len(df_current)} samples")
            
            if self.generated_data_path.exists():
                df_generated = pd.read_csv(self.generated_data_path)
                current_dfs.append(df_generated)
                logger.info(f"Loaded generated data: {len(df_generated)} samples")
            
            current_df = None
            if current_dfs:
                current_df = pd.concat(current_dfs, ignore_index=True)
                logger.info(f"Combined current data: {len(current_df)} samples")
            
            return reference_df, current_df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None, None
    
    def preprocess_data_for_comparison(self, df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
        """Preprocess data for drift comparison"""
        if df is None or df.empty:
            return df
        
        # Remove null values
        df = df.dropna(subset=['text'])
        
        # Sample data if too large
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        return df
    
    def jensen_shannon_drift(self, reference_features: np.ndarray, current_features: np.ndarray) -> Dict:
        """Calculate Jensen-Shannon divergence for drift detection"""
        try:
            # Compute mean feature vectors
            ref_mean = np.mean(reference_features, axis=0)
            cur_mean = np.mean(current_features, axis=0)
            
            # Normalize to probability distributions
            ref_dist = ref_mean / np.sum(ref_mean) if np.sum(ref_mean) > 0 else ref_mean
            cur_dist = cur_mean / np.sum(cur_mean) if np.sum(cur_mean) > 0 else cur_mean
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            ref_dist = ref_dist + epsilon
            cur_dist = cur_dist + epsilon
            
            # Calculate JS divergence
            js_distance = jensenshannon(ref_dist, cur_dist)
            
            return {
                'method': 'jensen_shannon',
                'distance': float(js_distance),
                'threshold': self.drift_thresholds['jensen_shannon'],
                'drift_detected': js_distance > self.drift_thresholds['jensen_shannon'],
                'severity': self.classify_drift_severity(js_distance, 'jensen_shannon')
            }
            
        except Exception as e:
            logger.error(f"Jensen-Shannon drift calculation failed: {e}")
            return {'method': 'jensen_shannon', 'error': str(e)}
    
    def kolmogorov_smirnov_drift(self, reference_features: np.ndarray, current_features: np.ndarray) -> Dict:
        """Kolmogorov-Smirnov test for drift detection"""
        try:
            # Flatten arrays for KS test
            ref_flat = reference_features.flatten()
            cur_flat = current_features.flatten()
            
            # Sample if too large
            if len(ref_flat) > 10000:
                ref_flat = np.random.choice(ref_flat, 10000, replace=False)
            if len(cur_flat) > 10000:
                cur_flat = np.random.choice(cur_flat, 10000, replace=False)
            
            # Perform KS test
            ks_statistic, p_value = ks_2samp(ref_flat, cur_flat)
            
            return {
                'method': 'kolmogorov_smirnov',
                'ks_statistic': float(ks_statistic),
                'p_value': float(p_value),
                'threshold': self.drift_thresholds['kolmogorov_smirnov'],
                'drift_detected': p_value < self.drift_thresholds['kolmogorov_smirnov'],
                'severity': self.classify_drift_severity(ks_statistic, 'kolmogorov_smirnov')
            }
            
        except Exception as e:
            logger.error(f"Kolmogorov-Smirnov drift calculation failed: {e}")
            return {'method': 'kolmogorov_smirnov', 'error': str(e)}
    
    def population_stability_index_drift(self, reference_features: np.ndarray, current_features: np.ndarray) -> Dict:
        """Population Stability Index for drift detection"""
        try:
            # Create bins based on reference data
            n_bins = 10
            
            # Use first feature for binning (or create composite feature)
            ref_values = reference_features[:, 0] if reference_features.ndim > 1 else reference_features
            cur_values = current_features[:, 0] if current_features.ndim > 1 else current_features
            
            # Create bins
            _, bin_edges = np.histogram(ref_values, bins=n_bins)
            
            # Calculate distributions
            ref_dist, _ = np.histogram(ref_values, bins=bin_edges)
            cur_dist, _ = np.histogram(cur_values, bins=bin_edges)
            
            # Convert to proportions
            ref_prop = ref_dist / np.sum(ref_dist)
            cur_prop = cur_dist / np.sum(cur_dist)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            ref_prop = ref_prop + epsilon
            cur_prop = cur_prop + epsilon
            
            # Calculate PSI
            psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
            
            return {
                'method': 'population_stability_index',
                'psi_score': float(psi),
                'threshold': self.drift_thresholds['population_stability_index'],
                'drift_detected': psi > self.drift_thresholds['population_stability_index'],
                'severity': self.classify_drift_severity(psi, 'population_stability_index')
            }
            
        except Exception as e:
            logger.error(f"PSI drift calculation failed: {e}")
            return {'method': 'population_stability_index', 'error': str(e)}
    
    def performance_drift(self, model, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict:
        """Detect performance drift by comparing model performance"""
        try:
            # Prepare data
            ref_X = reference_df['text'].values
            ref_y = reference_df['label'].values
            cur_X = current_df['text'].values
            cur_y = current_df['label'].values if 'label' in current_df.columns else None
            
            # Get predictions
            ref_pred = model.predict(ref_X)
            cur_pred = model.predict(cur_X)
            
            # Calculate performance metrics
            ref_accuracy = accuracy_score(ref_y, ref_pred)
            
            performance_metrics = {
                'reference_accuracy': float(ref_accuracy),
                'reference_samples': len(ref_X)
            }
            
            # If current data has labels, compare performance
            if cur_y is not None:
                cur_accuracy = accuracy_score(cur_y, cur_pred)
                performance_drop = ref_accuracy - cur_accuracy
                
                performance_metrics.update({
                    'current_accuracy': float(cur_accuracy),
                    'performance_drop': float(performance_drop),
                    'drift_detected': performance_drop > self.drift_thresholds['performance_degradation'],
                    'severity': self.classify_drift_severity(performance_drop, 'performance_degradation')
                })
            else:
                # Use prediction confidence as proxy
                ref_confidence = np.max(model.predict_proba(ref_X), axis=1)
                cur_confidence = np.max(model.predict_proba(cur_X), axis=1)
                
                confidence_drop = np.mean(ref_confidence) - np.mean(cur_confidence)
                
                performance_metrics.update({
                    'reference_confidence': float(np.mean(ref_confidence)),
                    'current_confidence': float(np.mean(cur_confidence)),
                    'confidence_drop': float(confidence_drop),
                    'drift_detected': confidence_drop > self.drift_thresholds['performance_degradation'],
                    'severity': self.classify_drift_severity(confidence_drop, 'performance_degradation')
                })
            
            return {
                'method': 'performance_drift',
                'threshold': self.drift_thresholds['performance_degradation'],
                **performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Performance drift calculation failed: {e}")
            return {'method': 'performance_drift', 'error': str(e)}
    
    def feature_importance_drift(self, model, reference_features: np.ndarray, current_features: np.ndarray) -> Dict:
        """Detect drift in feature importance"""
        try:
            # This is a simplified version - in practice, you'd compare feature importance
            # over time or use more sophisticated methods
            
            # Calculate feature statistics
            ref_mean = np.mean(reference_features, axis=0)
            cur_mean = np.mean(current_features, axis=0)
            
            # Calculate feature drift for each feature
            feature_drifts = np.abs(ref_mean - cur_mean) / (np.abs(ref_mean) + 1e-10)
            
            # Overall drift score
            overall_drift = np.mean(feature_drifts)
            max_drift = np.max(feature_drifts)
            
            return {
                'method': 'feature_importance_drift',
                'overall_drift': float(overall_drift),
                'max_feature_drift': float(max_drift),
                'threshold': self.drift_thresholds['feature_drift'],
                'drift_detected': overall_drift > self.drift_thresholds['feature_drift'],
                'severity': self.classify_drift_severity(overall_drift, 'feature_drift')
            }
            
        except Exception as e:
            logger.error(f"Feature importance drift calculation failed: {e}")
            return {'method': 'feature_importance_drift', 'error': str(e)}
    
    def statistical_distance_drift(self, reference_features: np.ndarray, current_features: np.ndarray) -> Dict:
        """Calculate various statistical distances for drift detection"""
        try:
            # Calculate means and covariances
            ref_mean = np.mean(reference_features, axis=0)
            cur_mean = np.mean(current_features, axis=0)
            
            # Euclidean distance between means
            euclidean_distance = np.linalg.norm(ref_mean - cur_mean)
            
            # Cosine similarity
            cosine_similarity = np.dot(ref_mean, cur_mean) / (np.linalg.norm(ref_mean) * np.linalg.norm(cur_mean))
            
            # Bhattacharyya distance (simplified)
            bhattacharyya_distance = -np.log(np.sum(np.sqrt(ref_mean * cur_mean)))
            
            return {
                'method': 'statistical_distance',
                'euclidean_distance': float(euclidean_distance),
                'cosine_similarity': float(cosine_similarity),
                'bhattacharyya_distance': float(bhattacharyya_distance),
                'drift_detected': euclidean_distance > self.drift_thresholds['feature_drift'],
                'severity': self.classify_drift_severity(euclidean_distance, 'feature_drift')
            }
            
        except Exception as e:
            logger.error(f"Statistical distance drift calculation failed: {e}")
            return {'method': 'statistical_distance', 'error': str(e)}
    
    def classify_drift_severity(self, score: float, method: str) -> str:
        """Classify drift severity based on score"""
        if score > self.alert_thresholds['high_drift']:
            return 'high'
        elif score > self.alert_thresholds['medium_drift']:
            return 'medium'
        elif score > self.alert_thresholds['low_drift']:
            return 'low'
        else:
            return 'none'
    
    def comprehensive_drift_detection(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict:
        """Perform comprehensive drift detection using multiple methods"""
        try:
            logger.info("Starting comprehensive drift detection...")
            
            # Load vectorizer and model
            vectorizer = self.load_vectorizer()
            model = self.load_model()
            
            if vectorizer is None:
                return {'error': 'Vectorizer not available'}
            
            # Prepare data
            reference_df = self.preprocess_data_for_comparison(reference_df, self.monitoring_config['max_samples'])
            current_df = self.preprocess_data_for_comparison(current_df, self.monitoring_config['max_samples'])
            
            if reference_df is None or current_df is None or len(reference_df) == 0 or len(current_df) == 0:
                return {'error': 'Insufficient data for drift detection'}
            
            # Vectorize text data
            ref_texts = reference_df['text'].tolist()
            cur_texts = current_df['text'].tolist()
            
            # Handle different vectorizer types
            if hasattr(vectorizer, 'transform'):
                ref_features = vectorizer.transform(ref_texts).toarray()
                cur_features = vectorizer.transform(cur_texts).toarray()
            else:
                return {'error': 'Vectorizer does not support transform method'}
            
            # Run all drift detection methods
            drift_results = {}
            
            # Feature-based drift detection
            for method_name in ['jensen_shannon', 'kolmogorov_smirnov', 'population_stability_index', 
                               'feature_importance_drift', 'statistical_distance']:
                try:
                    drift_results[method_name] = self.drift_methods[method_name](ref_features, cur_features)
                except Exception as e:
                    logger.error(f"Drift method {method_name} failed: {e}")
                    drift_results[method_name] = {'method': method_name, 'error': str(e)}
            
            # Performance-based drift detection
            if model is not None:
                try:
                    drift_results['performance_drift'] = self.performance_drift(model, reference_df, current_df)
                except Exception as e:
                    logger.error(f"Performance drift detection failed: {e}")
                    drift_results['performance_drift'] = {'method': 'performance_drift', 'error': str(e)}
            
            # Calculate overall drift score
            overall_drift = self.calculate_overall_drift_score(drift_results)
            
            # Create comprehensive report
            comprehensive_report = {
                'timestamp': datetime.now().isoformat(),
                'reference_samples': len(reference_df),
                'current_samples': len(current_df),
                'overall_drift_score': overall_drift['score'],
                'overall_drift_detected': overall_drift['detected'],
                'drift_severity': overall_drift['severity'],
                'individual_methods': drift_results,
                'recommendations': self.generate_drift_recommendations(drift_results, overall_drift)
            }
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"Comprehensive drift detection failed: {e}")
            return {'error': str(e)}
    
    def calculate_overall_drift_score(self, drift_results: Dict) -> Dict:
        """Calculate overall drift score from individual methods"""
        valid_scores = []
        detected_count = 0
        
        # Weight different methods
        method_weights = {
            'jensen_shannon': 0.3,
            'kolmogorov_smirnov': 0.2,
            'population_stability_index': 0.2,
            'performance_drift': 0.2,
            'feature_importance_drift': 0.05,
            'statistical_distance': 0.05
        }
        
        weighted_score = 0
        total_weight = 0
        
        for method, result in drift_results.items():
            if 'error' in result:
                continue
                
            # Extract score based on method
            if method == 'jensen_shannon':
                score = result.get('distance', 0)
            elif method == 'kolmogorov_smirnov':
                score = result.get('ks_statistic', 0)
            elif method == 'population_stability_index':
                score = result.get('psi_score', 0)
            elif method == 'performance_drift':
                score = result.get('performance_drop', result.get('confidence_drop', 0))
            else:
                score = result.get('overall_drift', 0)
            
            # Add to weighted score
            weight = method_weights.get(method, 0.1)
            weighted_score += score * weight
            total_weight += weight
            
            # Count detections
            if result.get('drift_detected', False):
                detected_count += 1
        
        # Calculate final score
        final_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine if drift is detected (majority vote with score consideration)
        drift_detected = (detected_count >= len(drift_results) / 2) or (final_score > 0.15)
        
        # Classify severity
        if final_score > 0.3:
            severity = 'high'
        elif final_score > 0.15:
            severity = 'medium'
        elif final_score > 0.05:
            severity = 'low'
        else:
            severity = 'none'
        
        return {
            'score': float(final_score),
            'detected': drift_detected,
            'severity': severity,
            'detection_count': detected_count,
            'total_methods': len(drift_results)
        }
    
    def generate_drift_recommendations(self, drift_results: Dict, overall_drift: Dict) -> List[str]:
        """Generate recommendations based on drift detection results"""
        recommendations = []
        
        if overall_drift['detected']:
            if overall_drift['severity'] == 'high':
                recommendations.extend([
                    "URGENT: High drift detected - immediate model retraining recommended",
                    "Consider switching to emergency backup model if available",
                    "Investigate data quality and collection processes"
                ])
            elif overall_drift['severity'] == 'medium':
                recommendations.extend([
                    "Moderate drift detected - schedule model retraining soon",
                    "Monitor performance metrics closely",
                    "Review recent data sources for quality issues"
                ])
            else:
                recommendations.extend([
                    "Low drift detected - increased monitoring recommended",
                    "Plan for model retraining in next cycle"
                ])
        
        # Method-specific recommendations
        for method, result in drift_results.items():
            if result.get('drift_detected', False):
                if method == 'performance_drift':
                    recommendations.append("Model performance degradation detected - prioritize retraining")
                elif method == 'jensen_shannon':
                    recommendations.append("Feature distribution drift detected - review data preprocessing")
                elif method == 'kolmogorov_smirnov':
                    recommendations.append("Statistical distribution change detected - validate data sources")
        
        return recommendations
    
    def save_drift_results(self, drift_results: Dict):
        """Save drift detection results to logs"""
        try:
            # Load existing logs
            logs = []
            if self.drift_log_path.exists():
                try:
                    with open(self.drift_log_path, 'r') as f:
                        logs = json.load(f)
                except:
                    logs = []
            
            # Add new results
            logs.append(drift_results)
            
            # Keep only last 1000 entries
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Save logs
            with open(self.drift_log_path, 'w') as f:
                json.dump(logs, f, indent=2)
            
            logger.info(f"Drift results saved to {self.drift_log_path}")
            
        except Exception as e:
            logger.error(f"Failed to save drift results: {e}")
    
    def monitor_drift(self) -> Optional[float]:
        """Main drift monitoring function"""
        try:
            logger.info("Starting drift monitoring...")
            
            # Load data
            reference_df, current_df = self.load_and_prepare_data()
            
            if reference_df is None or current_df is None:
                logger.warning("Insufficient data for drift monitoring")
                return None
            
            # Perform comprehensive drift detection
            drift_results = self.comprehensive_drift_detection(reference_df, current_df)
            
            if 'error' in drift_results:
                logger.error(f"Drift detection failed: {drift_results['error']}")
                return None
            
            # Save results
            self.save_drift_results(drift_results)
            
            # Log results
            overall_score = drift_results['overall_drift_score']
            severity = drift_results['drift_severity']
            
            logger.info(f"Drift monitoring completed")
            logger.info(f"Overall drift score: {overall_score:.4f}")
            logger.info(f"Drift severity: {severity}")
            
            if drift_results['overall_drift_detected']:
                logger.warning("DRIFT DETECTED!")
                for recommendation in drift_results['recommendations']:
                    logger.warning(f"Recommendation: {recommendation}")
            
            return overall_score
            
        except Exception as e:
            logger.error(f"Drift monitoring failed: {e}")
            return None

def monitor_drift():
    """Main function for external calls"""
    monitor = AdvancedDriftMonitor()
    return monitor.monitor_drift()

def main():
    """Main execution function"""
    monitor = AdvancedDriftMonitor()
    drift_score = monitor.monitor_drift()
    
    if drift_score is not None:
        print(f"✅ Drift monitoring completed. Score: {drift_score:.4f}")
    else:
        print("❌ Drift monitoring failed")
        exit(1)

if __name__ == "__main__":
    main()