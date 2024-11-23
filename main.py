import pandas as pd


def read_data() -> pd.DataFrame:
    df = pd.read_csv("books.csv", on_bad_lines="skip")
    df.rename(columns={"  num_pages": "num_pages"}, inplace=True)

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        [
            "authors",
            "num_pages",
            "ratings_count",
            "text_reviews_count",
            "publication_date",
            "publisher",
            "average_rating",
        ]
    ]

    # keep only the first author listed
    df.loc[:, "authors"] = df["authors"].apply(lambda d: d.split("/")[0])

    # use only the publication year
    df.loc[:, "publication_date"] = df["publication_date"].apply(
        lambda d: d.split("/")[2]
    )

    # one-hot encode authors and publishers
    df = pd.get_dummies(
        df,
        prefix=["author", "publisher"],
        columns=["authors", "publisher"],
        drop_first=True,
    )

    return df


if __name__ == "__main__":
    df = read_data()
    df = preprocess_data(df)
    print(df.head())
