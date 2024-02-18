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

# Get the top authors with the most rows
author_counts = df_selected['author_id'].value_counts()
top_authors = author_counts.head(1)
print("Top author_ids with the most rows:")
for author_id, count in top_authors.items():
    print(f"Author ID: {author_id}, Number of Rows: {count}")

# Filter the DataFrame to include only data from the top authors
num_rows = 100
top_authors_data = df_selected[df_selected['author_id'].isin(top_authors.index)]
top_authors_data = top_authors_data.head(num_rows)  # Limit to top rows

# Tokenize text and count frequencies for each document (ID) in top authors data
document_elements = []
for document, text in zip(top_authors_data['id'], top_authors_data['text']):
    # Split text into tokens using whitespace and punctuation as delimiters
    tokens = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
    word_counts = Counter(tokens)
    for element, frequency_in_document in word_counts.items():
        document_elements.append((document, element, float(frequency_in_document)))

# Create a DataFrame from the list of tuples
result_df = pd.DataFrame(document_elements, columns=['document', 'element', 'frequency_in_document'])

# Write the DataFrame to a new CSV file
result_df.to_csv('test_data/post_frequency.csv', index=False, mode='w', header=True)

print(f"CSV file 'post_frequency.csv' with only the top elements from top authors has been created successfully.")