import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import pandas as pd
import re


plt.rcParams["font.family"] = "serif"

# warning: garbage code
grade_to_emoji = {
    "U": '$\\circ$',
    "C": '$\\checkmark$',
    "I": '$\\times$'
}

def plot_inspect_samples_as_lasagne(data: pd.DataFrame, sample_col: str, index_col: str, grade_col: str, score_col: str, model_col: str, title_midfix: str,  score_type="ucat", mode="show", models=["openai/o1-2024-12-17", "openai/gpt-4o-2024-11-20", "anthropic/claude-3-5-sonnet-20241022", "cedar/CEDAR"]):
    """Lasagne plot Inspect samples so each row is a sample, each cell is a message, and the cell's color is the message's score.

    Args:
        data (pd.DataFrame): DataFrame of messages
        sample_col (str): column name of sample identifier
        index_col (str): column name of the message index (i.e. message number)
        grade_col (str): column name of the sample grade (NB expects "U", "C", "I" values)
        score_col (str): column name of used to colour the heatmap's cells
        model_col (str): column name of the model identifier
        title_midfix (str): string to be insterted between model name and score name in title
        score_type (str) : "ucat" for unordered categorical, "ocat" for ordered categorical, "cont" for continuous
        mode (str): "show" to display the plot, "save" to save it
        models (list): list of models plot sample for
    """
    
    
    data = data[data[model_col].isin(models)]
    max_index = data[index_col].max()
    
    original_data = data.copy()
    if score_type == "ucat":
        cat_to_code = {s: j for j, s in enumerate(data[score_col].unique())}
        code_to_cat = {j: s for s, j in cat_to_code.items()}
    elif score_type == "cont":
        cmap = plt.cm.turbo
        norm = plt.Normalize(original_data[score_col].min(), original_data[score_col].max())
    
        data = original_data[original_data[model_col] == model].copy()
    
        sample_lengths = data[sample_col].value_counts()
        data.sort_values(by=sample_col, key=lambda s_id: sample_lengths[s_id], inplace=True, ascending=False)
        
        fig, ax = plt.subplots(figsize=(11.7, 8.3)) # A4
        
        ax.yaxis.grid(False)
        ax.xaxis.grid(False)
        ax.set_facecolor('none')
        ax.set_title(title := f"{model}\n{title_midfix}\n{score_col}")
        for spine in ax.spines.values():
            spine.set_visible(False)
        sample_id_to_int = {sample_id: j for j, sample_id in enumerate(data[sample_col].unique())}
        sample_id_to_grade = {sample_id: grade for sample_id, grade in data[[sample_col, grade_col]].drop_duplicates().values}
        if score_type=="ucat":
            cmap = plt.cm.get_cmap("tab20", len(cat_to_code))
            norm = plt.Normalize(0, len(cat_to_code))
            data.apply(
                lambda row: ax.add_patch(patches.Rectangle((row[index_col], sample_id_to_int[row[sample_col]]), 1, 1, alpha=1, linewidth=0., edgecolor='black', facecolor=cmap(norm(cat_to_code[row[score_col]])), zorder=1)) if not pd.isna(row[score_col]) else "",
                axis=1
            )
            legend_elements = [Patch(facecolor=cmap(norm(code)), label=cat) 
                    for cat, code in cat_to_code.items()]
            plt.legend(handles=legend_elements, title='Categories')
        elif score_type=="ocat":
            raise NotImplemented
            cmap = plt.cm.inferno
            norm = plt.Normalize(0, len(data[score_col].unique())-1)
            data.apply(
                lambda row: ax.add_patch(patches.Rectangle((row[index_col], sample_id_to_int[row[sample_col]]), 1, 1, linewidth=0, edgecolor='none', facecolor=cmap(norm(cat_to_code[row[score_col]])), zorder=1)),
                axis=1
            )
            legend_elements = [Patch(facecolor=cmap(norm(code)), label=cat) for cat, code in cat_to_code.items()]
            plt.legend(handles=legend_elements, title='Categories')
        elif score_type=="cont":
            data.apply(
                lambda row: ax.add_patch(patches.Rectangle((row[index_col], sample_id_to_int[row[sample_col]]), 1, 1, linewidth=0, edgecolor='none', facecolor=cmap(norm(row[score_col])))),
                axis=1
            )
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
            cbar.set_label(score_col)
            
        else:
            raise ValueError(f"Unknown score_type: {score_type}")
        
        ax.set_ylim(0, max(sample_id_to_int.values()) + 1)
        ax.set_xlim(0, max_index)
        ax.set_xticks(np.arange(0.5, data[index_col].max() + 1.5, 10))
        ax.set_yticks(np.arange(0.5, max(sample_id_to_int.values()) + 1.5, 1))
        ax.set_xticklabels(np.arange(0, data[index_col].max() + 1, 10))
        ax.xaxis.set_ticks_position('top')  # Show ticks on top
        ax.xaxis.set_label_position('top')  # Show label on top
        ax.set_yticklabels([f"{k} {grade_to_emoji[sample_id_to_grade[k]]}" for k in sample_id_to_int.keys()])
        y_labels = ax.get_yticklabels()

        # Color individual tick labels
        for i, label in enumerate(y_labels):
            if "times" in label.get_text():  # Every other label
                label.set_color('red')
            elif "checkmark" in label.get_text():
                label.set_color('green')
            elif "circ" in label.get_text():
                label.set_color('grey')
            
        # Apply the modified labels back to the axes
        ax.set_yticklabels(y_labels)
        
        ax.set_ylabel("Sample ID")
        
            
        # Save or show
        plt.tight_layout()
        if mode == "save":
            filename =  re.sub('[^0-9a-zA-Z]+', '', title) + ".svg"
            plt.savefig(filename)
            plt.close()
        elif mode == "show":
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError(f"Unknown mode: {mode}")