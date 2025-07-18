import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from smolagents import (
    LocalPythonExecutor,
    OpenAIModel,
    ToolCallingAgent,
    tool,
)
from smolagents.local_python_executor import BASE_PYTHON_TOOLS

from mlebench.grade import validate_submission as validate_submission_fn
from mlebench.registry import registry

load_dotenv()

competition_name = "tabular-playground-series-may-2022"
competition = registry.get_competition(competition_name)

train_dataframe = pd.read_csv(competition.public_dir / "train.csv")
test_dataframe = pd.read_csv(competition.public_dir / "test.csv")


@tool
def validate_submission() -> str:
    """Checks that the submission is valid. Beware that this only checks that the file works, not that the predictions are correct! You should score the predictions yourself using train/test splits.

    Args:
        submission: The path to the submission file.
    """
    return validate_submission_fn(Path("submission.csv"), competition)[1]


model = OpenAIModel(model_id="o1")

instructions = """ADDITIONAL NOTES
------

- **Total Runtime**: You have a maximum of 24hours to attempt this task. You may iteratively develop your solution within this time frame, but bear in mind that program runtime also counts towards this limit.
- **Total Steps**: You have a maximum of 500 steps to submit your solution.
- You will be informed of the remaining time and steps at each step.
- **Use as much of the available time as possible to refine and optimize your submission**: Your goal is to achieve the best possible score, not just to submit your first working solution.
Therefore, while you are provided with a "final_answer" tool, you should not use it unless you are certain that no further improvements can be made.
Make sure to always:
- validate your answer using the "Validate Submission" tool (this only checks validity of the files, not that predictions are correct)
- manually spot-check predictions, building custom validation sets and grading functions
- compare different algorithms.
- generate plots of the data with matplotlib, export them under 'plot.png', then you will be able to view them.
You may keep working until the time limit expires -- as long as your best submission is stored at the designated location at the end of your run, it will be picked up for grading.
"""


@tool
def write_and_execute_code(code: str) -> str:
    """Writes and executes a given code script. This will return to you the print outputs of the code.

    Args:
        code: The code to write and execute. This code should be self-contained and should not rely on any external files except train.csv.
    """
    python_interpreter = LocalPythonExecutor(
        additional_authorized_imports=[
            "pandas.*",
            "numpy.*",
            "matplotlib.*",
            "seaborn.*",
            "scipy.*",
            "sklearn.*",
            "xgboost.*",
            # "lightgbm.*",
        ],
    )
    python_interpreter.send_tools(BASE_PYTHON_TOOLS)
    python_interpreter.state.update(
        {"train_dataframe": train_dataframe, "test_dataframe": test_dataframe}
    )
    output = python_interpreter(code)
    return output.logs


def plot_inspection_callback(memory_step, agent) -> str:
    """
    This callback is called after each step of the agent.
    It will add the plot under "plot.png", if there is any, to the agent's memory.
    """
    if "write_and_execute_code" in str(memory_step.tool_calls) and "plot.png" in str(
        memory_step.tool_calls
    ):
        if os.path.exists("plot.png"):
            with open("plot.png", "rb") as f:
                plot = f.read()
                memory_step.observations_images = [plot]
    return "Visualized plot.png"


agent = ToolCallingAgent(
    model=model,
    tools=[validate_submission, write_and_execute_code],
    instructions=instructions,
    step_callbacks=[plot_inspection_callback],
)

task = f"""You are trying to win a Kaggle competition. Your goal is to get the best possible score.
To this end, make good predictions for the table test_dataframe provided below, and write this submission under the `submission.csv` file.
To do so, generate and execute scripts that generate and test predictions.
Only when you are confident that you are getting good results, you can write the submission file under the `submission.csv` file.

# {competition.name}

{competition.description}

Sample submission file:
```csv
{competition.sample_submission.read_text()[:500]}

You can use the provided dataframe train_dataframe to train your model.
```

FYI, here are the top 5 scores of the leaderboard for this competition:
```
{pd.read_csv(competition.leaderboard)["score"].head(5)}
```
Do not return your final answer until you get a score comparable to the current leaderboard (sometimes lower is better), or if you have made at least 20 steps to try and improve your score.

Now go on!
"""


agent.run(
    task,
    additional_args={
        "train_dataframe": train_dataframe,
        "test_dataframe": test_dataframe,
    },
)
