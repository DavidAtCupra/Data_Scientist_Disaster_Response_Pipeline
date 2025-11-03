import sys
import pandas as pd
from sqlalchemy import create_engine

#python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
#python train_classifier.py ../data/DisasterResponse.db classifier.pkl

def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """Load and merge messages & categories datasets.

    Args:
        messages_filepath: Path to disaster_messages.csv
        categories_filepath: Path to disaster_categories.csv

    Returns:
        Merged DataFrame on common 'id' field.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge on 'id' (inner join keeps matched pairs only)
    df = messages.merge(categories, on="id", how="inner")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean merged dataset by expanding `categories` into binary columns.

    Steps:
      1) Split `categories` into separate columns.
      2) Extract category names for headers.
      3) Convert values to numeric 0/1.
      4) Concatenate back to the original dataframe and drop duplicates.
    """
    # Split the categories column
    categories = df["categories"].str.split(";", expand=True)

    # Use first row to extract new column names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames

    # Convert category values to 0/1 integers
    for column in categories.columns:
        categories[column] = categories[column].str.split("-").str[-1]
        categories[column] = pd.to_numeric(categories[column], errors="coerce").fillna(0).astype(int)
        # Some datasets contain values >1 (e.g., 'related' can be 2). Clip to {0,1}.
        categories[column] = categories[column].clip(upper=1)

    # Drop the original categories column and concat new ones
    df = df.drop(columns=["categories"]).join(categories)

    # Remove duplicates (by full row equality)
    df = df.drop_duplicates()

    # Optional: drop rows that have all category zeros (rarely informative)
    # df = df[(df[categories.columns].sum(axis=1) > 0)]

    return df


def save_data(df: pd.DataFrame, database_filename: str, table_name: str = "DisasterResponse") -> None:
    """Persist cleaned dataframe to a SQLite database.

    Args:
        df: Cleaned dataframe to save.
        database_filename: SQLite file path (e.g., 'data/DisasterResponse.db').
        table_name: Name of the table to write to (default: 'DisasterResponse').
    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql(table_name, engine, index=False, if_exists="replace")



def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print("Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "data/disaster_messages.csv data/disaster_categories.csv "
            "data/DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
