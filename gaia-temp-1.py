from inspect_ai import eval
from inspect_ai.solver import basic_agent
from inspect_ai.tool import bash
from inspect_evals.gaia import gaia

agent = basic_agent(tools=[bash()])
task = gaia(
    split="validation",
    instance_ids=[
        "cf106601-ab4f-4af9-b045-5295fe67b37d",
        "48eb8242-1099-4c26-95d4-ef22b002457a",
        "72c06643-a2fa-4186-aa5c-9ec33ae9b445",
    ],
)

# Validation dataset task IDs: https://huggingface.co/datasets/gaia-benchmark/GAIA/blob/main/2023/validation/metadata.jsonl
# Olympics question: cf106601-ab4f-4af9-b045-5295fe67b37d
# Nonindigenous crocodiles question: 48eb8242-1099-4c26-95d4-ef22b002457a (LEVEL 2)
# Marianas Trench question: 72c06643-a2fa-4186-aa5c-9ec33ae9b445"


# What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer.

# How many nonindigenous crocodiles were found in Florida from the year 2000 through 2020? You can get the data from the USGS Nonindigenous Aquatic Species database.sdfafdfdfdfffffffffffffffffffffffffff

# What is the volume in milliliters of a system comprised of 0.312 kg Freon-12 refrigerant when placed at the bottom of the Marianas Trench and allowed to stabilize at the Trench's peak temperature, rounded to the nearest mL? Provide your answer as just an integer value.

models = [
    "anthropic/claude-3-7-sonnet-20250219",
    "anthropic/claude-3-5-sonnet-20240620",
    "anthropic/claude-3-opus-20240229",
]
eval(
    task,
    model=models,
    temperature=0.0,
    max_connections=20,
    limit=None,
    epochs=1,
)
