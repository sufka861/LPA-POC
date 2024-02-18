import csv
import re
from collections import Counter

import pandas as pd

df = pd.read_csv('test_data/posts-dataset.csv')
df_selected = df[['id', 'author_id', 'text']]

grouped_by_author_id = df_selected.groupby('author_id')
# for author_id, group_df in grouped_by_author_id:
#     print(f"Author ID: {author_id}")
#     print(group_df)
#     print()

author_counts = df_selected['author_id'].value_counts()
top_authors = author_counts.head(1)
print("Top author_ids with the most rows:")
for author_id, count in top_authors.items():
    print(f"Author ID: {author_id}, Number of Rows: {count}")
top_authors_data = df_selected[df_selected['author_id'].isin(top_authors.index)]
# print(top_authors_data)



# Tokenize text and count frequencies for each document (ID)
word_counts = Counter()
for text in df['text']:
    # Split text into tokens using whitespace and punctuation as delimiters
    tokens = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
    word_counts.update(tokens)

# Select the top n most frequent words
n = 10
top_words = word_counts.most_common(n)
top_words_set = set(word for word, _ in top_words)

# Select only the required columns
df_selected = df[['id', 'text']]

# Tokenize text and count frequencies for each document (ID) for only the top n words
document_elements = []
for document, text in zip(df_selected['id'], df_selected['text']):
    tokens = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
    word_counts = Counter(tokens)
    for element, frequency_in_document in word_counts.items():
        if element in top_words_set:
            document_elements.append((document, element, float(frequency_in_document)))

# Create a DataFrame from the list of tuples
result_df = pd.DataFrame(document_elements, columns=['document', 'element', 'frequency_in_document'])

# Write the DataFrame to a new CSV file
result_df.to_csv('test_data/post_frequency.csv', index=False)
print(f"CSV file 'post_frequency.csv' with only the top {n} elements has been created successfully.")