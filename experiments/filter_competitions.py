import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("./experiments/competition_categories.csv")

    print(df["category"].unique())
    print(df.loc[df["competition_id"] == "spaceship-titanic"])

    df = df.loc[
        df["category"]
        .str.strip()
        .isin(
            [
                "Forecasting",
                "Tabular",
                "Signal Processing",
                # "Sequence to Sequence",
                # "Text (Other)",
                # "Text Classification",
            ]
        )
    ]

    df = df.loc[df["dataset_size_GB"] < 1]
    for id in df["competition_id"].unique():
        print(id)
    print("spaceship-titanic")
