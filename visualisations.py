import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import pandas as pd
from matplotlib.patches import Patch

# warning: garbage code
grade_to_emoji = {"U": "$\\circ$", "C": "$\\checkmark$", "I": "$\\times$"}


def lasagne_stacked_cont(
    data: pd.DataFrame,
    model: str,
    index_col: str = "index",
    score_col: str = "reasoning_n_chars",
    grade_col: str = "grade",
    model_col: str = "model",
    task_col: str = "task_name",
    fig = None,
    ax = None
):
    # The logic of this is that we get the colorbar and max index from the full dataset,
    # but only plot entries relating to the specified task and model
    max_index = data[index_col].max()
    cmap = plt.cm.get_cmap("viridis")
    norm = plt.Normalize(0., data[score_col].max())

    data = data[(data[model_col] == model)]
    data.sort_values(by=["task_name", "index"])
    task_to_id = {tn: j for j, tn in enumerate(data["task_name"].unique())}
    task_to_grade = {
        tn: g
        for tn, g in data.groupby("task_name").apply(lambda gdf: gdf["grade"].iloc[0]).items()
    }
    
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(figsize=(11.7, 2.3))

    grade = data[grade_col].iloc[0]

    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)

    data.apply(
        lambda row: ax.add_patch(
            patches.Rectangle(
                (int(row[index_col]), task_to_id[row["task_name"]]),
                1,
                1,
                alpha=1,
                linewidth=0.1,
                edgecolor="white",
                facecolor=cmap(norm(row[score_col])),
                zorder=1,
            )
        )
        if not pd.isna(row[score_col])
        else "",
        axis=1,
    )
    # sm = ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # Dummy array for the mappable
    # cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.1)
    # cbar.set_label(score_col)
    plt.legend()

    ax.set_title(f"{model}")
    ax.set_xlim(0, max_index)
    ax.set_ylim(0, max(task_to_id.values()) + 1)
    ax.set_ylabel("Task")
    ax.set_xlabel("Message #")
    ax.set_yticks([j + 0.5 for tn, j in task_to_id.items()])
    ax.set_yticklabels([f"{tn} - {grade_to_emoji[task_to_grade[tn]]}" for tn in task_to_id.keys()])

    plt.tight_layout()
    plt.show()


def lasagne_stacked(
    data: pd.DataFrame,
    model: str,
    index_col: str = "index",
    score_col: str = "tool_call",
    grade_col: str = "grade",
    model_col: str = "model",
    task_col: str = "task_name",
    fig = None,
    ax = None
):
    # The logic of this is that we get the colorbar and max index from the full dataset,
    # but only plot entries relating to the specified task and model
    max_index = data[index_col].max()
    cat_to_code = {s: j for j, s in enumerate(data[score_col].unique())}
    cmap = plt.cm.get_cmap("Set3", len(cat_to_code))
    norm = plt.Normalize(0, len(cat_to_code))

    data = data[(data[model_col] == model)]
    data.sort_values(by=["task_name", "index"])
    task_to_id = {tn: j for j, tn in enumerate(data["task_name"].unique())}
    task_to_grade = {
        tn: g
        for tn, g in data.groupby("task_name").apply(lambda gdf: gdf["grade"].iloc[0]).items()
    }
    
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(figsize=(11.7, 2.3))

    grade = data[grade_col].iloc[0]

    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)

    data.apply(
        lambda row: ax.add_patch(
            patches.Rectangle(
                (int(row[index_col]), task_to_id[row["task_name"]]),
                1,
                1,
                alpha=1,
                linewidth=0.1,
                edgecolor="white",
                facecolor=cmap(norm(cat_to_code[row[score_col]])),
                zorder=1,
            )
        )
        if not pd.isna(row[score_col])
        else "",
        axis=1,
    )
    legend_elements = [
        Patch(facecolor=cmap(norm(code)), label=cat)
        for cat, code in cat_to_code.items()
    ]
    fig.legend(handles=legend_elements, title="Categories", loc="outside right upper")

    ax.set_title(f"{model}")
    ax.set_xlim(0, max_index)
    ax.set_ylim(0, max(task_to_id.values()) + 1)
    ax.set_ylabel("Task")
    ax.set_xlabel("Message #")
    ax.set_yticks([j + 0.5 for tn, j in task_to_id.items()])
    ax.set_yticklabels([f"{task_to_id[tn]} - {grade_to_emoji[task_to_grade[tn]]}" for tn in task_to_id.keys()])

    plt.tight_layout()
    plt.show()

def lasagne_single(
    data: pd.DataFrame,
    model: str,
    task: str,
    index_col: str = "index",
    score_col: str = "tool_call",
    grade_col: str = "grade",
    model_col: str = "model",
    task_col: str = "task_name",
):
    # The logic of this is that we get the colorbar and max index from the full dataset,
    # but only plot entries relating to the specified task and model
    max_index = data[index_col].max()
    grade = data[grade_col].iloc[0]
    cat_to_code = {s: j for j, s in enumerate(data[score_col].unique())}
    cmap = plt.cm.get_cmap("Set3", len(cat_to_code))
    norm = plt.Normalize(0, len(cat_to_code))

    data = data[(data[model_col] == model) & (data[task_col] == task)]
    data.sort_values(by=["index"])

    assert len(data[model_col].unique()) == 1
    assert len(data[task_col].unique()) == 1

    fig, ax = plt.subplots(figsize=(11.7, 1.3)) 
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)

    data.apply(
        lambda row: ax.add_patch(
            patches.Rectangle(
                (row[index_col], 0.0),
                1,
                1,
                alpha=1,
                linewidth=0.1,
                edgecolor="black",
                facecolor=cmap(norm(cat_to_code[row[score_col]])),
                zorder=1,
            )
        )
        if not pd.isna(row[score_col])
        else "",
        axis=1,
    )
    legend_elements = [
        Patch(facecolor=cmap(norm(code)), label=cat)
        for cat, code in cat_to_code.items()
    ]
    fig.legend(handles=legend_elements, title="Categories", loc="outside right upper")

    ax.set_title(f"{model}\n{task}\n{grade}")
    ax.set_xlim(0, max_index)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_yticklabels("")

    plt.tight_layout()
    plt.show()
