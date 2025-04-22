from pickle import loads
from uuid import uuid4

import duckdb

# inspect-evaldb --log_dir /home/ubuntu/los-alamos/logs --db_uri output.db --hide_manual
con = duckdb.connect("gaia-t0.db")

try:
    df = (
        con.execute("""
        SELECT * FROM raw_eval_log_headers
    """)
        .df()
        .assign(eval_log=lambda df: df["pickled_evallog"].apply(lambda pel: loads(pel)))
        .assign(
            status=lambda df: df["eval_log"].apply(lambda el: el.status),
            eval_run_id=lambda df: df["eval_log"].apply(lambda el: el.eval.run_id),
            eval_task_id=lambda df: df["eval_log"].apply(lambda el: el.eval.task_id),
            eval_task=lambda df: df["eval_log"].apply(lambda el: el.eval.task),
            model=lambda df: df["eval_log"].apply(lambda el: el.eval.model),
        )
        .drop(columns=["pickled_evallog", "eval_log"])
        .assign(raw_eval_log_header_uuid=lambda df: df["uuid"])
        .assign(uuid=lambda df: df["uuid"].apply(lambda _: uuid4()))
        .drop(["inserted"], axis=1)
    )
    con.execute("CREATE TABLE tidy_eval_log_headers AS SELECT * FROM df")
except duckdb.duckdb.CatalogException:
    pass

df = con.execute("""
SELECT * 
FROM tidy_eval_messages tem
JOIN tidy_eval_sample_headers tesh ON tesh.raw_log_uuid = tem.raw_log_uuid
JOIN tidy_eval_log_headers telh ON telh.raw_eval_log_header_uuid = tem.raw_log_uuid
""").df()
df["task_name"] = df["target"].apply(
    lambda v: {"55": "trench", "6": "crocodiles", "CUB": "olympics"}[v]
)

df.groupby(["model", "task_name"]).apply(lambda gdf: gdf.shape[0], include_groups=False)

df.loc[
    (df["task_name"] == "olympics")
    & (df["model"] == "anthropic/claude-3-opus-20240229")
]

breakpoint()

# Want to join the tidy messages and tidy headers
# Can do this via the raw log uuid
# Join each separately, return the result
# tidy_eval_sample_headers.raw_log_uuid
# tidy_eval_messages.raw_log_uuid
# tidy_eval_log_headers.raw_eval_log_header_uuid

# Message counts for each sample
