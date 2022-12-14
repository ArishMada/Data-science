import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
from sklearn.feature_extraction.text import CountVectorizer

books = pd.read_csv("Books.csv", low_memory=False)
# print(books.head(3))

ratings = pd.read_csv("Ratings.csv")
# print(ratings.head(10))

# Preprocessing
books_data = books.merge(ratings, on="ISBN")  # add the ratings data to the book data using the ISBN
# print(books_data.head())

df = books_data.copy()
df.dropna(inplace=True)
df.drop(columns=["ISBN", "Year-Of-Publication", "Image-URL-S", "Image-URL-M"], axis=1, inplace=True)
df.drop(index=df[df["Book-Rating"] == 0].index, inplace=True)
df.reset_index(drop=True, inplace=True)
df["Book-Title"] = df["Book-Title"].apply(lambda x: re.sub("[\W_]+", " ", x).strip())
print(df.head())

new_df = df[df['User-ID'].map(df['User-ID'].value_counts()) > 200]  # Drop users who vote less than 200 times.
users_pivot = new_df.pivot_table(index=["User-ID"],columns=["Book-Title"],values="Book-Rating")
users_pivot.fillna(0, inplace=True)


def content_based(bookTitle):
    bookTitle = str(bookTitle)

    if bookTitle in df["Book-Title"].values:
        rating_count = pd.DataFrame(df["Book-Title"].value_counts())
        rare_books = rating_count[rating_count["Book-Title"] <= 200].index
        common_books = df[~df["Book-Title"].isin(rare_books)]

        if bookTitle in rare_books:
            most_common = pd.Series(common_books["Book-Title"].unique()).sample(3).values
            print("No Recommendations for this Book\n ")
            print("YOU MAY TRY: \n ")
            print("{}".format(most_common[0]), "\n")
            print("{}".format(most_common[1]), "\n")
            print("{}".format(most_common[2]), "\n")
        else:
            common_books = common_books.drop_duplicates(subset=["Book-Title"])
            common_books.reset_index(inplace=True)
            common_books["index"] = [i for i in range(common_books.shape[0])]
            targets = ["Book-Title", "Book-Author", "Publisher"]
            common_books["all_features"] = [" ".join(common_books[targets].iloc[i,].values) for i in
                                            range(common_books[targets].shape[0])]
            vectorizer = CountVectorizer()
            common_booksVector = vectorizer.fit_transform(common_books["all_features"])
            similarity = cosine_similarity(common_booksVector)
            index = common_books[common_books["Book-Title"] == bookTitle]["index"].values[0]
            similar_books = list(enumerate(similarity[index]))
            similar_booksSorted = sorted(similar_books, key=lambda x: x[1], reverse=True)[1:6]
            books = []
            for i in range(len(similar_booksSorted)):
                books.append(common_books[common_books["index"] == similar_booksSorted[i][0]]["Book-Title"].item())

            for i in range(len(books)):
                print("RATING: {}".format(round(df[df["Book-Title"] == books[i]]["Book-Rating"].mean(), 1)),
                      "TITLE: {}".format(common_books.loc[common_books["Book-Title"] == books[i],
                                                          "Book-Title"][:1].values[0]))

    else:
        print("COULD NOT FIND")


content_based("The Chamber")
print("\n")
content_based("Airframe")
print("\n")
content_based("Wild Animus")
print("\n")
content_based(",Talking Donkeys and Wheels of Fire Bible Stories That Are Truly Bizarre")
print("\n")
content_based("Harry Potter and the Order of the Phoenix Book 5")
print("\n")
content_based("Life of Pi")