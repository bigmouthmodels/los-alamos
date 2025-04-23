import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "serif"

# warning: garbage code
grade_to_emoji = {"U": "$\\circ$", "C": "$\\checkmark$", "I": "$\\times$"}


def plot_messages_as_lasagne(
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
    cat_to_code = {s: j for j, s in enumerate(data[score_col].unique())}
    cmap = plt.cm.get_cmap("tab20", len(cat_to_code))
    norm = plt.Normalize(0, len(cat_to_code))

    data = data[(data[model_col] == model) & (data[task_col] == task)]

    assert len(data[model_col].unique()) == 1
    assert len(data[task_col].unique()) == 1

    fig, ax = plt.subplots(figsize=(11.7, 1.3))  # A4

    grade = data[grade_col].iloc[0]

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
                linewidth=0.0,
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
    plt.legend(handles=legend_elements, title="Categories", loc="right")

    ax.set_title(f"{model}\n{task}\n{grade}")
    ax.set_xlim(0, max_index)
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_yticklabels("")

    plt.tight_layout()
    plt.show()
