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

competition = registry.get_competition("ventilator-pressure-prediction")

train_dataframe = pd.read_csv(competition.public_dir / "train.csv")
test_dataframe = pd.read_csv(competition.public_dir / "test.csv")


@tool
def validate_submission() -> str:
    """Checks that the submission is valid.

    Args:
        submission: The path to the submission file.
    """
    return validate_submission_fn(Path("submission.csv"), competition)[1]


model = OpenAIModel(model_id="gpt-4.1")

instructions = """ADDITIONAL NOTES
------

- **Total Runtime**: You have a maximum of 24hours to attempt this task. You may iteratively develop your solution within this time frame, but bear in mind that program runtime also counts towards this limit.
- **Total Steps**: You have a maximum of 500 steps to submit your solution.
- You will be informed of the remaining time and steps at each step.
- **Use as much of the available time as possible to refine and optimize your submission**: Your goal is to achieve the best possible score, not just to submit your first working solution.
Therefore, while you are provided with a "final_answer" tool, you should not use it unless you are certain that no further improvements can be made. Consider strategies like validating your answer using the "Validate Submission" tool, manually spot-checking predictions, building custom validation sets and grading functions, and comparing different algorithms. You may keep working until the time limit expires -- as long as your best submission is stored at the designated location at the end of your run, it will be picked up for grading.
You can generate plots of the data under file 'plot.png', then you will be able to view them
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
        ],
    )
    python_interpreter.send_tools(BASE_PYTHON_TOOLS)
    python_interpreter.state.update(
        {"train_dataframe": train_dataframe, "test_dataframe": test_dataframe}
    )
    output = python_interpreter(code)
    return output.logs


agent = ToolCallingAgent(
    model=model,
    tools=[validate_submission, write_and_execute_code],
    instructions=instructions,
)


task = f"""You are given a Kaggle competition, your goal is to make predictions for the table test_dataframe provided below, and write this submission under the `submission.csv` file.
To do so, generate and execute a full script that will produce the submission file.

# {competition.name}

{competition.description}

Sample submission file:
```csv
{competition.sample_submission.read_text()[:1000]}

You can use the provided dataframe train_dataframe to train your model.
```


Now go on!
"""


agent.run(
    task,
    additional_args={
        "train_dataframe": train_dataframe,
        "test_dataframe": test_dataframe,
    },
)
