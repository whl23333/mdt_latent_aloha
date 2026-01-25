"""
Analyze the linear relationship between latent motion embeddings and cumulative actions.

This script:
1. Loads dataset with latent_static, latent_gripper, and actions
2. Computes cumulative actions over first 5 frames:
   - For X,Y,Z,Roll,Pitch,Yaw (dims 0-5): Sum of delta values
   - For Gripper (dim 6): Last state (-1 or 1, not a delta)
3. Extracts 256-dim embeddings from first 8 tokens of latent motion tokenizer
4. Performs linear regression for each of 7 action dimensions
5. Visualizes scatter plots with R² scores
6. Generates statistical report
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from tqdm import tqdm
import pandas as pd

# Add mdt_latent_aloha to path
sys.path.insert(0, '/home/hlwang/mdt_policy_reproduce/mdt_latent_aloha')

from mdt.datasets.hulc_data_module import HulcDataModule
from mdt.models.mdt_agent import MDTAgent


class LatentActionAnalyzer:
    """Analyzer for latent embedding vs cumulative action regression."""
    
    def __init__(
        self,
        config_path: str,
        dataset_path: str,
        output_dir: str = "./analysis_output",
        num_samples: int = 2000,
        sample_plot_size: int = 500,
        device: str = "cuda:0"
    ):
        """
        Args:
            config_path: Path to training config.yaml
            dataset_path: Path to dataset root directory
            output_dir: Directory to save outputs
            num_samples: Total number of samples to collect from dataset
            sample_plot_size: Number of points to visualize in scatter plots
            device: Device to run on
        """
        self.config_path = config_path
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        self.sample_plot_size = sample_plot_size
        self.device = device
        
        # Action dimension names for plotting
        self.action_dims = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Gripper (state)']
        
        # Storage for collected data
        self.embeddings = []
        self.cumulative_actions = []
        
        # Regression results
        self.models: List[LinearRegression] = []
        self.metrics: Dict[str, List[float]] = {
            'r2': [], 'mse': [], 'mae': [], 'coef_norm': []
        }
        
        print(f"Initializing analyzer...")
        print(f"Config: {config_path}")
        print(f"Dataset: {dataset_path}")
        print(f"Output: {output_dir}")
        
    def load_model_and_data(self):
        """Load MDT model and dataset."""
        print("\n=== Loading Model and Dataset ===")
        
        # Use Hydra to properly load config with defaults composition
        config_dir = str(Path(self.config_path).parent.absolute())
        config_name = Path(self.config_path).stem
        
        print(f"Loading config with Hydra composition...")
        print(f"  Config dir: {config_dir}")
        print(f"  Config name: {config_name}")
        
        with initialize_config_dir(config_dir=config_dir):
            cfg = compose(config_name=config_name)
        
        # Override paths
        cfg.root_data_dir = self.dataset_path
        cfg.datamodule.root_data_dir = self.dataset_path
        
        # Initialize datamodule
        print("Initializing datamodule...")
        self.datamodule: HulcDataModule = hydra.utils.instantiate(cfg.datamodule)
        self.datamodule.prepare_data()
        self.datamodule.setup()
        
        # Get validation dataset
        val_dataloader = self.datamodule.val_dataloader()
        # Handle CombinedLoader (newer PyTorch Lightning) or dict
        if hasattr(val_dataloader, 'loaders'):
            # CombinedLoader case - get 'lang' or 'vis' loader
            loader_key = 'lang' if 'lang' in val_dataloader.loaders else 'vis'
            loader = val_dataloader.loaders[loader_key]
            # Handle CycleIterator wrapper
            if hasattr(loader, 'loader'):
                loader = loader.loader
            elif hasattr(loader, '_loader'):
                loader = loader._loader
            self.dataset = loader.dataset
        elif isinstance(val_dataloader, dict):
            # Dict case
            loader_key = 'lang' if 'lang' in val_dataloader else 'vis'
            self.dataset = val_dataloader[loader_key].dataset
        else:
            # Single loader case
            self.dataset = val_dataloader.dataset
        print(f"Dataset size: {len(self.dataset)}")
        
        # Initialize MDT agent from config (will auto-load LMT)
        print("Initializing MDT agent with latent motion tokenizer...")
        # Remove ckpt_path to prevent loading full model weights
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        if 'ckpt_path' in model_cfg:
            model_cfg['ckpt_path'] = None
        model_cfg = OmegaConf.create(model_cfg)
        
        self.model = hydra.utils.instantiate(model_cfg)
        self.model.eval()
        self.model.to(self.device)
        
        # Verify LMT is loaded
        assert self.model.lmt is not None, "Latent motion tokenizer not found in model!"
        assert self.model.predict_embeddings, "Model must have predict_embeddings=True"
        self.model.lmt.eval()
        print(f"✓ MDT Agent initialized with LMT (query_num={self.model.lmt.m_former.query_num})")
        print(f"  Latent motion view: {self.model.latent_motion_view}")
        print(f"  Latent action num: {self.model.latent_action_num}")
        
    def collect_data(self):
        """Collect embeddings and cumulative actions from dataset."""
        print(f"\n=== Collecting {self.num_samples} samples ===")
        
        collected = 0
        pbar = tqdm(total=self.num_samples, desc="Collecting data")
        
        # Sample random indices from dataset
        indices = np.random.choice(len(self.dataset), size=min(self.num_samples, len(self.dataset)), replace=False)
        
        with torch.no_grad():
            for idx in indices:
                if collected >= self.num_samples:
                    break
                    
                try:
                    # Get dataset batch
                    sample = self.dataset[int(idx)]
                    
                    # Check if latent data exists
                    if 'latent_static' not in sample or sample['latent_static'] is None:
                        continue
                    
                    # Extract data
                    latent_static = sample['latent_static'].unsqueeze(0).to(self.device)  # [1, T, C, H, W]
                    latent_gripper = sample['latent_gripper'].unsqueeze(0).to(self.device)
                    actions = sample['actions'].unsqueeze(0)  # [1, 10, 7]
                    
                    # Compute cumulative action over first 5 frames
                    # For dimensions 0-5 (x,y,z,roll,pitch,yaw): sum delta values
                    # For dimension 6 (gripper): take last state (it's -1/1, not a delta)
                    cumulative_action = torch.zeros(1, 7)
                    cumulative_action[:, :6] = actions[:, :5, :6].sum(dim=1)  # Sum deltas
                    cumulative_action[:, 6] = actions[:, 4, 6]  # Last gripper state
                    cumulative_action = cumulative_action.cpu().numpy()  # [1, 7]
                    
                    # Prepare dataset_batch dict for compute_embedding_targets
                    sample['latent_action_diff'] = [sample['latent_action_diff'].tolist()]
                    sample['latent_action_num'] = [sample['latent_action_num'].tolist()]
                    dataset_batch = {
                        'latent_static': latent_static,
                        'latent_gripper': latent_gripper,
                        'latent_action_diff': sample['latent_action_diff'],
                        'latent_action_num': sample['latent_action_num'],
                    }
                    
                    # Get embeddings using model's method
                    embeddings = self.model.compute_embedding_targets(dataset_batch)  # [1, 2*8, 32]
                    
                    # Extract first 8 tokens and flatten to 256-dim
                    embedding_vec = embeddings[:, :8, :].reshape(1, -1).cpu().numpy()  # [1, 256]
                    
                    # Store
                    self.embeddings.append(embedding_vec[0])  # [256]
                    self.cumulative_actions.append(cumulative_action[0])  # [7]
                    
                    collected += 1
                    pbar.update(1)
                    
                except Exception as e:
                    # Skip problematic samples
                    continue
        
        pbar.close()
        
        # Convert to numpy arrays
        self.embeddings = np.array(self.embeddings)  # [N, 256]
        self.cumulative_actions = np.array(self.cumulative_actions)  # [N, 7]
        
        print(f"✓ Collected {len(self.embeddings)} valid samples")
        print(f"  Embeddings shape: {self.embeddings.shape}")
        print(f"  Cumulative actions shape: {self.cumulative_actions.shape}")
        print(f"  Cumulative action stats:")
        for i, dim_name in enumerate(self.action_dims):
            print(f"    {dim_name}: mean={self.cumulative_actions[:, i].mean():.4f}, "
                  f"std={self.cumulative_actions[:, i].std():.4f}")
    
    def train_regressions(self):
        """Train linear regression for each action dimension."""
        print("\n=== Training Linear Regressions ===")
        
        X = self.embeddings  # [N, 256]
        
        for i, dim_name in enumerate(self.action_dims):
            y = self.cumulative_actions[:, i]  # [N]
            
            # Train linear regression
            model = LinearRegression()
            model.fit(X, y)
            self.models.append(model)
            
            # Compute metrics
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            coef_norm = np.linalg.norm(model.coef_)
            
            self.metrics['r2'].append(r2)
            self.metrics['mse'].append(mse)
            self.metrics['mae'].append(mae)
            self.metrics['coef_norm'].append(coef_norm)
            
            print(f"{dim_name:8s}: R²={r2:.4f}, MSE={mse:.6f}, MAE={mae:.6f}, ||coef||={coef_norm:.4f}")
    
    def visualize_scatter_plots(self):
        """Create scatter plots for each action dimension."""
        print("\n=== Creating Scatter Plots ===")
        
        # Sample subset for visualization
        n_samples = min(self.sample_plot_size, len(self.embeddings))
        sample_indices = np.random.choice(len(self.embeddings), size=n_samples, replace=False)
        
        X_sample = self.embeddings[sample_indices]
        y_sample = self.cumulative_actions[sample_indices]
        
        # Create 7 subplots
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, (dim_name, model) in enumerate(zip(self.action_dims, self.models)):
            ax = axes[i]
            
            # Get predictions for sampled data
            y_true = y_sample[:, i]
            y_pred = model.predict(X_sample)
            
            # Compute R² for sample
            r2 = r2_score(y_true, y_pred)
            
            # Compute errors for color coding
            errors = np.abs(y_true - y_pred)
            
            # Scatter plot
            scatter = ax.scatter(y_true, y_pred, c=errors, cmap='coolwarm', 
                               alpha=0.6, s=20, edgecolors='k', linewidths=0.5)
            
            # Add y=x reference line
            lims = [
                min(y_true.min(), y_pred.min()),
                max(y_true.max(), y_pred.max())
            ]
            ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Perfect fit')
            
            # Add regression line (should be close to y=x if R² is high)
            ax.plot(lims, lims, 'r-', alpha=0.3, linewidth=1.5, label='Predicted')
            
            # Labels and title
            ax.set_xlabel(f'True Cumulative {dim_name}', fontsize=11)
            ax.set_ylabel(f'Predicted Cumulative {dim_name}', fontsize=11)
            ax.set_title(f'{dim_name}: R² = {r2:.4f}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Absolute Error')
        
        # Remove extra subplot
        fig.delaxes(axes[7])
        
        plt.tight_layout()
        plot_path = self.output_dir / 'regression_scatter_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved scatter plots to {plot_path}")
        plt.close()
    
    def visualize_coefficient_heatmap(self):
        """Visualize regression coefficient matrix as heatmap."""
        print("\n=== Creating Coefficient Heatmap ===")
        
        # Stack all coefficients: [7, 256]
        coef_matrix = np.array([model.coef_ for model in self.models])
        
        # Reshape to [7, 8, 32] - 8 codes, each 32-dim
        coef_reshaped = coef_matrix.reshape(7, 8, 32)
        
        # Compute L2 norm for each code: [7, 8]
        # This shows the "strength" of each code's contribution to each action
        code_importance = np.linalg.norm(coef_reshaped, axis=2)
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(20, 10))
        
        # 1. Main heatmap: 8 codes × 7 actions (transposed for better readability)
        ax1 = plt.subplot(2, 1, 1)
        im1 = ax1.imshow(code_importance, aspect='auto', cmap='YlOrRd')
        
        # Labels
        ax1.set_yticks(range(7))
        ax1.set_yticklabels(self.action_dims, fontsize=11)
        ax1.set_xticks(range(8))
        ax1.set_xticklabels([f'Code {i+1}' for i in range(8)], fontsize=11)
        ax1.set_xlabel('Latent Motion Code (8 codes from codebook)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Action Dimension', fontsize=12, fontweight='bold')
        ax1.set_title('Code Importance: L2 Norm of 32-dim Coefficients per Code', 
                     fontsize=14, fontweight='bold')
        
        # Add values on heatmap
        for i in range(7):
            for j in range(8):
                text = ax1.text(j, i, f'{code_importance[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        # Colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Coefficient L2 Norm', fontsize=11)
        
        # 2. Detailed view: Show coefficient distribution for each code
        ax2 = plt.subplot(2, 1, 2)
        
        # Create a blocked heatmap showing all 256 dims with visible separations
        im2 = ax2.imshow(coef_matrix, aspect='auto', cmap='RdBu_r',
                        vmin=-np.abs(coef_matrix).max(),
                        vmax=np.abs(coef_matrix).max())
        
        # Add vertical lines to separate codes
        for i in range(1, 8):
            ax2.axvline(x=i*32-0.5, color='black', linewidth=2)
        
        # Labels
        ax2.set_yticks(range(7))
        ax2.set_yticklabels(self.action_dims, fontsize=11)
        ax2.set_xlabel('Embedding Dimension (256 = 8 codes × 32 dims)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Action Dimension', fontsize=12, fontweight='bold')
        ax2.set_title('Detailed Coefficients: All 256 Dimensions (separated by code blocks)',
                     fontsize=14, fontweight='bold')
        
        # Add code labels on top
        for i in range(8):
            ax2.text(i*32 + 16, -0.8, f'Code {i+1}', ha='center', fontsize=10,
                    fontweight='bold', color='darkblue')
        
        # Colorbar
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Coefficient Value', fontsize=11)
        
        plt.tight_layout()
        heatmap_path = self.output_dir / 'coefficient_heatmap.png'
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved coefficient heatmap to {heatmap_path}")
        plt.close()
        
        # Also save code importance matrix as CSV for easy analysis
        code_importance_df = pd.DataFrame(
            code_importance,
            index=self.action_dims,
            columns=[f'Code_{i+1}' for i in range(8)]
        )
        csv_path = self.output_dir / 'code_importance.csv'
        code_importance_df.to_csv(csv_path)
        print(f"✓ Saved code importance matrix to {csv_path}")
        
        return code_importance
    
    def generate_report(self):
        """Generate markdown report with statistics."""
        print("\n=== Generating Report ===")
        
        report_lines = [
            "# Latent Motion Embedding vs Cumulative Action Regression Analysis\n",
            f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            f"**Dataset:** {self.dataset_path}\n",
            f"**Config:** {self.config_path}\n",
            f"**Samples Collected:** {len(self.embeddings)}\n",
            "\n## Summary\n",
            "This analysis evaluates the linear predictability of cumulative actions (sum of first 5 frames) ",
            "from 256-dimensional latent motion embeddings (first 8 tokens × 32 dimensions).\n",
            "\n## Results by Action Dimension\n",
            "| Dimension | R² Score | MSE | MAE | Coef. Norm | Predictability |\n",
            "|-----------|----------|-----|-----|------------|----------------|\n"
        ]
        
        # Sort by R² descending
        sorted_indices = np.argsort(self.metrics['r2'])[::-1]
        
        for rank, idx in enumerate(sorted_indices, 1):
            dim_name = self.action_dims[idx]
            r2 = self.metrics['r2'][idx]
            mse = self.metrics['mse'][idx]
            mae = self.metrics['mae'][idx]
            coef_norm = self.metrics['coef_norm'][idx]
            
            # Categorize predictability
            if r2 > 0.7:
                pred_category = "🟢 High"
            elif r2 > 0.4:
                pred_category = "🟡 Medium"
            elif r2 > 0.2:
                pred_category = "🟠 Low"
            else:
                pred_category = "🔴 Very Low"
            
            report_lines.append(
                f"| **{dim_name}** | {r2:.4f} | {mse:.6f} | {mae:.6f} | {coef_norm:.4f} | {pred_category} |\n"
            )
        
        # Overall statistics
        mean_r2 = np.mean(self.metrics['r2'])
        median_r2 = np.median(self.metrics['r2'])
        
        report_lines.extend([
            "\n## Overall Statistics\n",
            f"- **Mean R²:** {mean_r2:.4f}\n",
            f"- **Median R²:** {median_r2:.4f}\n",
            f"- **Best Dimension:** {self.action_dims[sorted_indices[0]]} (R²={self.metrics['r2'][sorted_indices[0]]:.4f})\n",
            f"- **Worst Dimension:** {self.action_dims[sorted_indices[-1]]} (R²={self.metrics['r2'][sorted_indices[-1]]:.4f})\n",
            "\n## Interpretation\n",
            "- **R² > 0.5**: Strong linear relationship, embeddings capture this action component well\n",
            "- **0.2 < R² < 0.5**: Moderate relationship, some predictive power but nonlinear factors may exist\n",
            "- **R² < 0.2**: Weak relationship, embeddings do not linearly encode this action dimension\n",
            "\n## Cumulative Action Statistics\n",
            "| Dimension | Mean | Std | Min | Max |\n",
            "|-----------|------|-----|-----|-----|\n"
        ])
        
        for i, dim_name in enumerate(self.action_dims):
            mean_val = self.cumulative_actions[:, i].mean()
            std_val = self.cumulative_actions[:, i].std()
            min_val = self.cumulative_actions[:, i].min()
            max_val = self.cumulative_actions[:, i].max()
            report_lines.append(
                f"| {dim_name} | {mean_val:.4f} | {std_val:.4f} | {min_val:.4f} | {max_val:.4f} |\n"
            )
        
        report_lines.extend([
            "\n## Visualizations\n",
            "- **Scatter Plots:** [regression_scatter_plots.png](regression_scatter_plots.png)\n",
            "- **Coefficient Heatmap:** [coefficient_heatmap.png](coefficient_heatmap.png)\n",
            "  - Top panel: 8×7 matrix showing L2 norm of each code's contribution\n",
            "  - Bottom panel: Full 256×7 matrix with code blocks separated\n",
            "- **Code Importance Matrix:** [code_importance.csv](code_importance.csv)\n",
            "\n## Code-Level Analysis\n",
            "The 256-dim embeddings are structured as 8 codes × 32 dimensions from a learned codebook.\n",
            "Each code represents a discrete motion pattern. The heatmap shows which codes are most\n",
            "important for predicting each action dimension.\n",
            "\n## Files Generated\n",
            f"- Regression models saved to: `{self.output_dir}`\n",
            "- Scatter plots: `regression_scatter_plots.png`\n",
            "- Coefficient heatmap: `coefficient_heatmap.png`\n",
            "- Code importance matrix: `code_importance.csv`\n",
            "- This report: `analysis_report.md`\n"
        ])
        
        # Write report
        report_path = self.output_dir / 'analysis_report.md'
        with open(report_path, 'w') as f:
            f.writelines(report_lines)
        
        print(f"✓ Saved report to {report_path}")
    
    def save_models(self):
        """Save trained regression models."""
        print("\n=== Saving Models ===")
        
        import pickle
        
        models_data = {
            'models': self.models,
            'metrics': self.metrics,
            'action_dims': self.action_dims,
            'embeddings_shape': self.embeddings.shape,
            'cumulative_actions_shape': self.cumulative_actions.shape
        }
        
        models_path = self.output_dir / 'regression_models.pkl'
        with open(models_path, 'wb') as f:
            pickle.dump(models_data, f)
        
        print(f"✓ Saved models to {models_path}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "="*70)
        print("LATENT MOTION EMBEDDING → CUMULATIVE ACTION REGRESSION ANALYSIS")
        print("="*70)
        
        # Step 1: Load model and data
        self.load_model_and_data()
        
        # Step 2: Collect data
        self.collect_data()
        
        # Step 3: Train regressions
        self.train_regressions()
        
        # Step 4: Visualize
        self.visualize_scatter_plots()
        self.visualize_coefficient_heatmap()
        
        # Step 5: Generate report
        self.generate_report()
        
        # Step 6: Save models
        self.save_models()
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print(f"📂 All outputs saved to: {self.output_dir.absolute()}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze latent embedding vs action regression")
    parser.add_argument('--config', type=str, required=True, help='Path to training config.yaml')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--output', type=str, default='./latent_action_analysis', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=2000, help='Number of samples to collect')
    parser.add_argument('--plot_samples', type=int, default=500, help='Number of points in scatter plots')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on')
    
    args = parser.parse_args()
    
    analyzer = LatentActionAnalyzer(
        config_path=args.config,
        dataset_path=args.dataset,
        output_dir=args.output,
        num_samples=args.num_samples,
        sample_plot_size=args.plot_samples,
        device=args.device
    )
    
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
