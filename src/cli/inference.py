#!/usr/bin/env python
"""
Flexible PALM Inference Script
Supports CSV, FASTA, or direct sequence input
Provides sequence-level predictions (CSV) and residue-level predictions (JSON)
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# Import PALM modules - these must be installed in the environment
try:
    from src.helpers.dataset import CSVDataLoader
    from src.model.composite_model import CompositeModel
except ImportError as e:
    print("Error: Cannot import PALM modules.")
    print("Please ensure PALM is properly installed in your environment:")
    print("  1. Activate your conda/poetry environment with PALM installed")
    print("  2. Or install PALM: pip install -e /path/to/PALM")
    print(f"\nDetailed error: {e}")
    import sys
    sys.exit(1)

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


class FlexiblePALMInference:
    """Flexible inference handler for PALM models"""
    
    def __init__(self, models_dir='./PALM_models', 
                 data_root='./datasets/',
                 use_cuda=True):
        """
        Initialize inference handler
        
        Args:
            models_dir: Path to PALM_models directory containing PALM, PALM_NNK, PALM_NNK_OH subdirectories
            data_root: Data root folder for PALM config
            use_cuda: Whether to use CUDA if available
        """
        self.models_dir = Path(models_dir)
        self.data_root = data_root
        
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        print(f"Using device: {self.device}")
        print(f"Models directory: {self.models_dir}")
        
        if self.device.type == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Verify models directory exists
        if not self.models_dir.exists():
            raise ValueError(f"Models directory not found: {self.models_dir}")
        
        # List available models
        self.available_models = self._discover_models()
        if self.available_models:
            print(f"Available models: {', '.join(self.available_models)}")
        else:
            print("Warning: No models found in the models directory")
    
    def _discover_models(self):
        """Discover available model variants in the models directory"""
        models = []
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and any(model_dir.glob('fold*/model.yaml')):
                models.append(model_dir.name)
        return sorted(models)
    
    def prepare_input_data(self, input_data, input_type='csv'):
        """
        Prepare input data from various sources
        
        Args:
            input_data: Path to file or list of sequences
            input_type: 'csv', 'fasta', or 'sequences'
        
        Returns:
            DataFrame with sequences and names
        """
        if input_type == 'csv':
            df = pd.read_csv(input_data)
            if 'sequence' not in df.columns:
                raise ValueError("CSV must contain 'sequence' column")
            if 'name' not in df.columns:
                df['name'] = [f"seq_{i}" for i in range(len(df))]
                
        elif input_type == 'fasta':
            try:
                from Bio import SeqIO
            except ImportError:
                print("Error: BioPython is required for FASTA input")
                print("Install with: pip install biopython")
                raise
                
            sequences = []
            names = []
            for record in SeqIO.parse(input_data, "fasta"):
                sequences.append(str(record.seq))
                names.append(record.id)
            df = pd.DataFrame({'sequence': sequences, 'name': names})
            
        elif input_type == 'sequences':
            # Direct sequence input
            if isinstance(input_data, str):
                sequences = [input_data]
            else:
                sequences = input_data
            names = [f"seq_{i}" for i in range(len(sequences))]
            df = pd.DataFrame({'sequence': sequences, 'name': names})
            
        else:
            raise ValueError(f"Unknown input type: {input_type}")
        
        # Add required columns for PALM compatibility
        if 'value_bool' not in df.columns:
            df['value_bool'] = 1  # Dummy value
        if 'dataset' not in df.columns:
            df['dataset'] = 'user_input'
        if 'data_split' not in df.columns:
            df['data_split'] = 'test'
        if 'len' not in df.columns:
            df['len'] = [len(seq) for seq in df['sequence']]
        
        return df
    
    def load_model(self, model_path):
        """Load a single model from a fold directory"""
    
        model_path = Path(model_path).resolve()
        print(f"Loading model from: {model_path}")
        
        # Verify required files exist
        model_yaml = model_path / 'model.yaml'
        model_state = model_path / 'model_state_dict.pt'
        
        if not model_yaml.exists():
            raise FileNotFoundError(f"model.yaml not found in {model_path}")
        if not model_state.exists():
            raise FileNotFoundError(f"model_state_dict.pt not found in {model_path}")
        
        with initialize_config_dir(config_dir=str(model_path), version_base=None, job_name=""):
            cfg = compose(
                config_name='model',
                overrides=[
                    f"+general.composite_model_path={str(model_path)}",
                    "general.run_mode=test",
		    "persistence.data_root_folder=."
                ]
            )
            
            # Initialize model in inference-only mode which now sets it to true thus dataload is none 
            model = CompositeModel(cfg, inference_only=True)
            
            if hasattr(model, 'predictor') and hasattr(model.predictor, 'model'):
                model.predictor.model = model.predictor.model.to(self.device)
    
        return model, cfg
    
    def run_single_model_inference(self, model, cfg, df):
        """
        Run inference with a single model
        
        Returns:
            Dictionary with sequence and residue predictions
        """
        # Prepare data
        data_split_column = cfg.dataset.data_split_column
        df[data_split_column] = 'test'
        dataloader = CSVDataLoader(cfg, df)
        
        # Run forward pass
        predictions = model.forward(dataloader)
        
        # Get sequence-level predictions - handle both tensor and MaskedArray
        if hasattr(predictions.predictions_probability, 'cpu'):
            sequence_predictions = predictions.predictions_probability.cpu().numpy()
        else:
            sequence_predictions = np.array(predictions.predictions_probability)
        
        # Get residue-level predictions
        if hasattr(model.predictor.model.o_unflattened, 'cpu'):
            output_numpy = model.predictor.model.o_unflattened.cpu().numpy()
        else:
            output_numpy = np.array(model.predictor.model.o_unflattened)
        
        # Process residue predictions for each sequence
        residue_predictions = []
        for j in range(len(df)):
            seq_len = len(df.iloc[j]['sequence'])
            residue_predictions.append(output_numpy[j, :seq_len])
        
        return {
            'sequence_predictions': sequence_predictions,
            'residue_predictions': residue_predictions
        }
    
    def run_inference(self, input_data, input_type='csv', model_name='PALM', 
                     ensemble=True, output_prefix='predictions'):
        """
        Run inference with automatic model discovery
        
        Args:
            input_data: Input data (file path or sequences)
            input_type: Type of input ('csv', 'fasta', 'sequences')
            model_name: Model variant to use (PALM, PALM_NNK, PALM_NNK_OH)
            ensemble: Whether to use ensemble predictions
            output_prefix: Prefix for output files (will create .csv and .json)
        
        Returns:
            Tuple of (sequence_df, residue_dict)
        """
        # Validate model name
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {', '.join(self.available_models)}")
        
        # Prepare input data
        df = self.prepare_input_data(input_data, input_type)
        print(f"\nProcessing {len(df)} sequences with model: {model_name}")
        
        # Get model directory
        model_dir = self.models_dir / model_name
        
        # Find available folds
        fold_dirs = sorted(model_dir.glob('fold*'))
        if not fold_dirs:
            raise ValueError(f"No fold directories found in {model_dir}")
        
        print(f"Found {len(fold_dirs)} folds")
        
        # Initialize storage for predictions
        all_seq_preds = []
        residue_predictions_dict = {}
        
        # Initialize residue predictions dictionary structure
        for idx, row in df.iterrows():
            residue_predictions_dict[row['name']] = {
                'sequence': row['sequence'],
                'length': len(row['sequence'])
            }
        
        # Run inference for each fold
        successful_folds = 0
        for fold_path in fold_dirs:
            fold_name = fold_path.name
            print(f"\nProcessing {fold_name}...")
            
            # try:
            model, cfg = self.load_model(fold_path)
            results = self.run_single_model_inference(model, cfg, df)
            
            all_seq_preds.append(results['sequence_predictions'])
            successful_folds += 1
            
            # Add fold predictions to dataframe (sequence-level)
            df[f'{model_name}_{fold_name}_seq_score'] = results['sequence_predictions']
            
            # Store residue predictions in dictionary
            for idx, (_, row) in enumerate(df.iterrows()):
                protein_name = row['name']
                residue_predictions_dict[protein_name][f'{model_name}_{fold_name}_residue_scores'] = \
                    results['residue_predictions'][idx].tolist()
                
            # except Exception as e:
            #     print(f"Warning: Error processing {fold_name}: {e}")
            #     continue
        
        if successful_folds == 0:
            raise RuntimeError("No models successfully processed")
        
        print(f"\nSuccessfully processed {successful_folds}/{len(fold_dirs)} folds")
        
        # Calculate ensemble predictions if requested
        if ensemble and len(all_seq_preds) > 1:
            # Sequence-level ensemble
            ensemble_seq_pred = np.mean(all_seq_preds, axis=0)
            df[f'{model_name}_ensemble_seq_score'] = ensemble_seq_pred
            
            # Residue-level ensemble
            for protein_name in residue_predictions_dict.keys():
                fold_scores = []
                for fold_path in fold_dirs:
                    fold_name = fold_path.name
                    key = f'{model_name}_{fold_name}_residue_scores'
                    if key in residue_predictions_dict[protein_name]:
                        fold_scores.append(residue_predictions_dict[protein_name][key])
                
                if fold_scores:
                    ensemble_res = np.mean(fold_scores, axis=0).tolist()
                    residue_predictions_dict[protein_name][f'{model_name}_ensemble_residue_scores'] = ensemble_res
                    
                    # Add summary statistics
                    residue_predictions_dict[protein_name]['summary'] = {
                        'mean_residue_score': float(np.mean(ensemble_res)),
                        'max_residue_score': float(np.max(ensemble_res)),
                        'min_residue_score': float(np.min(ensemble_res)),
                        'high_risk_positions': [int(i) for i in np.where(np.array(ensemble_res) > 0.5)[0]]
                    }
            
            print(f"Ensemble predictions calculated from {successful_folds} folds")
        
        # Prepare sequence-level output dataframe
        seq_columns = ['name', 'sequence']
        pred_columns = [col for col in df.columns if 'seq_score' in col]
        seq_columns.extend(pred_columns)
        sequence_df = df[seq_columns].copy()
        
        # Save outputs
        csv_file = f"{output_prefix}_sequences.csv"
        json_file = f"{output_prefix}_residues.json"
        
        # Save sequence predictions to CSV
        sequence_df.to_csv(csv_file, index=False)
        print(f"\nSequence predictions saved to: {csv_file}")
        
        # Save residue predictions to JSON
        with open(json_file, 'w') as f:
            json.dump(residue_predictions_dict, f, cls=NumpyEncoder, indent=2)
        print(f"Residue predictions saved to: {json_file}")
        
        # Print summary
        self.print_summary(sequence_df, residue_predictions_dict, ensemble)
        
        return sequence_df, residue_predictions_dict
    
    def print_summary(self, seq_df, res_dict, ensemble):
        """Print summary of predictions"""
        print(f"\n{'='*60}")
        print("PREDICTION SUMMARY")
        print(f"{'='*60}")
        
        for _, row in seq_df.iterrows():
            protein_name = row['name']
            print(f"\nProtein: {protein_name}")
            print(f"  Sequence length: {len(row['sequence'])}")
            
            # Sequence-level predictions
            seq_pred_cols = [col for col in seq_df.columns if 'seq_score' in col]
            for col in seq_pred_cols:
                print(f"  {col}: {row[col]:.4f}")
            
            # Residue-level summary (if ensemble)
            if ensemble and 'summary' in res_dict[protein_name]:
                summary = res_dict[protein_name]['summary']
                print(f"  Residue statistics:")
                print(f"    Mean score: {summary['mean_residue_score']:.4f}")
                print(f"    Max score: {summary['max_residue_score']:.4f}")
                print(f"    Min score: {summary['min_residue_score']:.4f}")
                
                high_risk = summary['high_risk_positions']
                if high_risk:
                    display_positions = high_risk[:10]
                    suffix = "..." if len(high_risk) > 10 else ""
                    print(f"    High-risk positions (>0.5): {display_positions}{suffix}")
                    print(f"    Total high-risk positions: {len(high_risk)}")
                else:
                    print(f"    High-risk positions (>0.5): None")


def main():
    parser = argparse.ArgumentParser(description='Flexible PALM inference script')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--csv', type=str, help='CSV file with sequences')
    input_group.add_argument('--fasta', type=str, help='FASTA file')
    input_group.add_argument('--sequences', type=str, nargs='+', help='Direct sequence input')
    
    # Model options
    parser.add_argument('--model_name', type=str, default='PALM',
                       help='Model variant to use (check available models in PALM_models directory)')
    parser.add_argument('--ensemble', action='store_true', default=True,
                       help='Use ensemble predictions (default: True)')
    parser.add_argument('--no_ensemble', dest='ensemble', action='store_false',
                       help='Disable ensemble predictions')
    
    # Output options
    parser.add_argument('--output_prefix', type=str, default='predictions',
                       help='Prefix for output files (creates _sequences.csv and _residues.json)')
    
    # Path options
    parser.add_argument('--models_dir', type=str, default='./PALM_models',
                       help='Directory containing PALM model variants')
    parser.add_argument('--data_root', type=str, default='./datasets/',
                       help='Data root folder for PALM config')
    
    # Other options
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage even if CUDA is available')
    parser.add_argument('--list_models', action='store_true',
                       help='List available models and exit')
    
    args = parser.parse_args()
    
    # Initialize inference handler
    inference = FlexiblePALMInference(
        models_dir=args.models_dir,
        data_root=args.data_root,
        use_cuda=not args.cpu
    )
    
    # List models and exit if requested
    if args.list_models:
        print("\nAvailable models:")
        for model in inference.available_models:
            model_path = Path(args.models_dir) / model
            fold_count = len(list(model_path.glob('fold*')))
            print(f"  - {model} ({fold_count} folds)")
        return
    
    # Determine input type and data
    if args.csv:
        input_data = args.csv
        input_type = 'csv'
    elif args.fasta:
        input_data = args.fasta
        input_type = 'fasta'
    else:
        input_data = args.sequences
        input_type = 'sequences'
    
    # Run inference
    # try:
    seq_results, res_results = inference.run_inference(
        input_data=input_data,
        input_type=input_type,
        model_name=args.model_name,
        ensemble=args.ensemble,
        output_prefix=args.output_prefix
    )
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    print(f"Output files created:")
    print(f"  📊 {args.output_prefix}_sequences.csv - Sequence-level predictions")
    print(f"  📈 {args.output_prefix}_residues.json - Residue-level predictions")
        
    # except Exception as e:
    #     print(f"\nError during inference: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return 1


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
