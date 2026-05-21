import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

# Apply dark theme aesthetic
plt.style.use('dark_background')
sns.set_style("darkgrid", {"axes.facecolor": "#121212", "figure.facecolor": "#121212", "grid.color": "#2c2c2c"})
sns.set_context("talk")

class Visualizer:
    """Handles all visual output generation using a modern dark aesthetic."""
    def __init__(self, output_dir: str = "assets"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Custom neon colors
        self.colors = {"cyan": "#00FFFF", "magenta": "#FF00FF", "lime": "#39FF14", "yellow": "#FFFF00"}

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap="cool", cbar=False, 
                         linewidths=0.5, linecolor='gray')
        plt.title(f'Confusion Matrix: {title}', color=self.colors['cyan'], pad=20)
        plt.xlabel('Predicted Label', color=self.colors['lime'])
        plt.ylabel('True Label', color=self.colors['lime'])
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved confusion matrix to {filepath}")

    def plot_roc_curve(self, y_true: np.ndarray, y_scores_dl: np.ndarray, y_scores_auto: np.ndarray, filename: str):
        plt.figure(figsize=(10, 8))
        
        # DL Curve
        fpr_dl, tpr_dl, _ = roc_curve(y_true, y_scores_dl)
        auc_dl = auc(fpr_dl, tpr_dl)
        plt.plot(fpr_dl, tpr_dl, color=self.colors['cyan'], lw=2, label=f'Deep Learning (AUC = {auc_dl:.2f})')
        
        # Automata Curve
        fpr_auto, tpr_auto, _ = roc_curve(y_true, y_scores_auto)
        auc_auto = auc(fpr_auto, tpr_auto)
        plt.plot(fpr_auto, tpr_auto, color=self.colors['magenta'], lw=2, label=f'Automata (AUC = {auc_auto:.2f})')
        
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        
        plt.title('ROC Curve: DL vs Automata', color=self.colors['yellow'], pad=20)
        plt.xlabel('False Positive Rate', color='white')
        plt.ylabel('True Positive Rate', color='white')
        plt.legend(loc="lower right", frameon=True, facecolor="#1a1a1a", edgecolor="white")
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved ROC curve to {filepath}")

    def plot_transition_heatmap(self, transition_matrix: dict, filename: str):
        # Convert nested dict to dataframe
        states = list(transition_matrix.keys())
        # To avoid massive plots, take top 20 states
        if len(states) > 20:
            states = states[:20]
            
        df = pd.DataFrame(index=states, columns=states).fillna(0.0)
        for s1 in states:
            if s1 in transition_matrix:
                for s2, prob in transition_matrix[s1].items():
                    if s2 in states:
                        df.loc[s1, s2] = prob
                        
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.astype(float), cmap="viridis", annot=False, cbar=True)
        plt.title('Transition Probability Heatmap (Top 20 States)', color=self.colors['cyan'], pad=20)
        plt.xlabel('Next State', color=self.colors['lime'])
        plt.ylabel('Current State', color=self.colors['lime'])
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved Transition Heatmap to {filepath}")

    def plot_parameter_sensitivity(self, df_params: pd.DataFrame, filename: str):
        plt.figure(figsize=(10, 6))
        
        # Plot lines for each alphabet size
        for a in df_params['alphabet_size'].unique():
            subset = df_params[df_params['alphabet_size'] == a]
            plt.plot(subset['window_size'], subset['f1_score'], marker='o', lw=2, 
                     label=f'Alphabet Size: {a}')
            
        plt.title('Automata Parameter Sensitivity', color=self.colors['magenta'], pad=20)
        plt.xlabel('Window Size', color='white')
        plt.ylabel('F1 Score', color='white')
        plt.legend(frameon=True, facecolor="#1a1a1a", edgecolor="white")
        plt.grid(True, color="#2c2c2c")
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved Parameter Sensitivity graph to {filepath}")

    def plot_automata_state_diagram(self, transition_matrix: dict, filename: str):
        import networkx as nx
        G = nx.DiGraph()
        
        states = list(transition_matrix.keys())
        if len(states) > 20:
            states = states[:20]
            
        for s1 in states:
            if s1 in transition_matrix:
                for s2, prob in transition_matrix[s1].items():
                    if s2 in states and prob > 0.05: # only draw significant edges
                        G.add_edge(s1, s2, weight=prob)
                        
        plt.figure(figsize=(14, 12))
        pos = nx.spring_layout(G, seed=42, k=0.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=self.colors['cyan'], 
                               node_size=2000, alpha=0.8, edgecolors='white')
                               
        # Draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] * 5 for u, v in edges] # scale thickness
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, 
                               edge_color=self.colors['magenta'], 
                               arrowsize=20, alpha=0.6, connectionstyle='arc3,rad=0.1')
                               
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", 
                                font_color="black", font_weight="bold")
                                
        # Draw edge labels (probabilities)
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                     font_color=self.colors['lime'], font_size=8)
                                     
        plt.title('Automata State Diagram (Top 20 States)', color=self.colors['yellow'], pad=20)
        plt.axis('off')
        
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300, facecolor='#121212')
        plt.close()
        print(f"Saved Automata State Diagram to {filepath}")

