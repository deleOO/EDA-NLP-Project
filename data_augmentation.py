from textattack.augmentation import EasyDataAugmenter
import pandas as pd

def augment_data(df: pd.DataFrame, pct_words_to_swap: float = 0.1, transformations_per_example: int = 4) -> pd.DataFrame:
    """
    This function takes in a dataframe containing text data and target column, applies EDA augmentation to the text data and returns a new dataframe with augmented and shuffled data
    :param df: Dataframe containing text data and target column
    :param pct_words_to_swap: percentage of words to swap in each text
    :param transformations_per_example: number of augmented text to create for each original text
    :return: Dataframe with augmented and shuffled data
    """
    # Create the EasyDataAugmenter object with the desired parameters
    eda_aug = EasyDataAugmenter(pct_words_to_swap=pct_words_to_swap, transformations_per_example=transformations_per_example)

    augmented_tweets = []

    # Iterate over each tweet in the dataframe
    for _, row in df.iterrows():
        text = row["text"]

        for augmented_text in eda_aug.augment(text):
            augmented_tweets.append([augmented_text, row["target"]])

    # Create a new dataframe with the augmented tweets
    augmented_df = pd.DataFrame(augmented_tweets, columns=["text", "target"])

    # Concatenate the original and augmented dataframes
    df_augmented = pd.concat([df, augmented_df], ignore_index=True)

    # shuffle the data
    df_augmented = df_augmented.sample(frac=1).reset_index(drop=True)

    return df_augmented