#!/usr/bin/env python
"""
Flexible PALM Inference Script
Supports CSV, FASTA, or direct sequence input
Provides sequence-level predictions (CSV) and residue-level predictions (JSON)
Optional interactive residue plots with Plotly
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    
    def create_residue_plots_plotly(self, proteins_data, output_file=None):
        """
        Create interactive residue aggregation plots for multiple proteins in a single HTML file
        
        Args:
            proteins_data: Dictionary with protein data including sequences and scores
            output_file: Output HTML file path (if None, auto-generated)
        
        Returns:
            Path to saved HTML file
        """

        n_proteins = len(proteins_data)
        
        # Create subplots - arrange in a grid

        cols = min(2, n_proteins)  # Max 2 columns
        rows = (n_proteins + cols - 1) // cols  # Calculate required rows
        
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=list(proteins_data.keys()),
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Track colors for consistent fold coloring across proteins
        fold_colors = {}
        color_palette = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for idx, (protein_name, data) in enumerate(proteins_data.items()):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            
            sequence = data['sequence']
            seq_length = len(sequence)
            residue_numbers = np.arange(1, seq_length + 1)
            
            # Add scatter plots for each fold
            for fold_idx, (fold_name, scores) in enumerate(data['fold_scores'].items()):
                scores = np.array(scores)
                
                # Assign consistent color to each fold
                if fold_name not in fold_colors:
                    fold_colors[fold_name] = color_palette[len(fold_colors) % len(color_palette)]
                
                # Only show legend for first protein
                show_legend = (idx == 0)
                
                fig.add_trace(
                    go.Scatter(
                        x=residue_numbers,
                        y=scores,
                        mode='markers',
                        marker=dict(
                            size=3,
                            opacity=0.6,
                            color=fold_colors[fold_name]
                        ),
                        name=fold_name,
                        showlegend=show_legend,
                        legendgroup=fold_name,
                        hovertemplate='<b>Protein:</b> ' + protein_name + '<br>' +
                                     '<b>Fold:</b> ' + fold_name + '<br>' +
                                     '<b>Position:</b> %{x}<br>' +
                                     '<b>Score:</b> %{y:.3f}<br>' +
                                     '<b>Residue:</b> %{text}<br>' +
                                     '<extra></extra>',
                        text=[sequence[i] for i in range(seq_length)]
                    ),
                    row=row, col=col
                )
            
            # Add ensemble line plot if available
            if data['ensemble_scores'] is not None:
                ensemble_scores = np.array(data['ensemble_scores'])
                
                fig.add_trace(
                    go.Scatter(
                        x=residue_numbers,
                        y=ensemble_scores,
                        mode='lines',
                        line=dict(color='black', width=2),
                        name='Ensemble',
                        showlegend=(idx == 0),
                        legendgroup='ensemble',
                        hovertemplate='<b>Protein:</b> ' + protein_name + '<br>' +
                                     '<b>Ensemble</b><br>' +
                                     '<b>Position:</b> %{x}<br>' +
                                     '<b>Score:</b> %{y:.3f}<br>' +
                                     '<b>Residue:</b> %{text}<br>' +
                                     '<extra></extra>',
                        text=[sequence[i] for i in range(seq_length)]
                    ),
                    row=row, col=col
                )
            
            # Update x-axis for this subplot with residue labels
            seq_length = len(sequence)
            
            #its tough to show tick labels so wrote a soft edge case for this

            # Create tick labels with residue number and amino acid
            if seq_length <= 50:
                # Show all residues for short sequences
                tickvals = list(range(1, seq_length + 1))
                ticktext = [f"{i}<br>{sequence[i-1]}" for i in tickvals]
            elif seq_length <= 100:
                # Show every other residue for medium sequences
                step = 2
                tickvals = list(range(1, seq_length + 1, step))
                ticktext = [f"{i}<br>{sequence[i-1]}" for i in tickvals]
            else:
                # Show every nth residue for long sequences
                step = max(1, seq_length // 25)
                tickvals = list(range(1, seq_length + 1, step))
                ticktext = [f"{i}<br>{sequence[i-1]}" for i in tickvals]
            
            fig.update_xaxes(
                title_text='Residue number',
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext,
                showgrid=True,
                gridcolor='lightgray',
                row=row, col=col
            )
            
            # Update y-axis for this subplot
            fig.update_yaxes(
                title_text='Aggregation score',
                showgrid=True,
                gridcolor='lightgray',
                row=row, col=col
            )
        
        # Update overall layout
        fig.update_layout(
            title=dict(
                text='Residue Aggregation Scores - All Proteins',
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            plot_bgcolor='white',
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="right",
                x=0.98
            ),
            height=max(400, 300 * rows),  # Adaptive height based on number of rows
            showlegend=True
        )
        
        # Save to HTML
        if output_file is None:
            output_file = "residue_plots_all_peptides.html"
        
        fig.write_html(output_file)
        print(f"Interactive plots for all sequences saved to: {output_file}")
        
        return output_file
    
    def run_inference(self, input_data, input_type='csv', model_name='PALM', 
                     ensemble=True, output_prefix='predictions',
                     create_plots=False, plot_format='plotly'):
        """
        Run inference with automatic model discovery
        
        Args:
            input_data: Input data (file path or sequences)
            input_type: Type of input ('csv', 'fasta', 'sequences')
            model_name: Model variant to use (PALM, PALM_NNK, PALM_NNK_OH)
            ensemble: Whether to use ensemble predictions
            output_prefix: Prefix for output files (will create .csv and .json)
            create_plots: Whether to create residue plots
            plot_format: Format for plots ('plotly' for interactive HTML)
        
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
        
        # Create plots if requested
        if create_plots:
            print("\nGenerating residue plots...")
            
            # Prepare data for all proteins
            proteins_plot_data = {}
            
            for protein_name, data in residue_predictions_dict.items():
                sequence = data['sequence']
                
                # Collect all fold scores
                fold_scores = {}
                for key in data.keys():
                    if 'residue_scores' in key and 'ensemble' not in key:
                        # Extract fold name from key
                        fold_name = key.replace(f'{model_name}_', '').replace('_residue_scores', '')
                        fold_scores[fold_name] = data[key]
                
                # Get ensemble scores if available
                ensemble_scores = None
                if f'{model_name}_ensemble_residue_scores' in data:
                    ensemble_scores = data[f'{model_name}_ensemble_residue_scores']
                
                proteins_plot_data[protein_name] = {
                    'sequence': sequence,
                    'fold_scores': fold_scores,
                    'ensemble_scores': ensemble_scores
                }
            
            # Create single HTML file with all plots
            if plot_format.lower() == 'plotly':
                plot_file = f"{output_prefix}_residue_plots_all.html"
                self.create_residue_plots_plotly(
                    proteins_data=proteins_plot_data,
                    output_file=plot_file
                )
                print(f"\nGenerated single interactive plot file: {plot_file}")
        
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
    
    # Plot options
    parser.add_argument('--plot', action='store_true',
                       help='Generate interactive residue plots (HTML)')
    parser.add_argument('--plot_format', type=str, default='plotly',
                       choices=['plotly'],
                       help='Plot format (default: plotly for interactive HTML)')
    
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
    seq_results, res_results = inference.run_inference(
        input_data=input_data,
        input_type=input_type,
        model_name=args.model_name,
        ensemble=args.ensemble,
        output_prefix=args.output_prefix,
        create_plots=args.plot,
        plot_format=args.plot_format
    )
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print("="*60)
    print(f"Output files created:")
    print(f"  📊 {args.output_prefix}_sequences.csv - Sequence-level predictions")
    print(f"  📈 {args.output_prefix}_residues.json - Residue-level predictions")
    
    if args.plot:
        print(f"  🎨 {args.output_prefix}_residue_plots_all.html - Interactive plots for all proteins")


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)