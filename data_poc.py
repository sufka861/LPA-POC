import pandas as pd
import re
from collections import Counter

# Reading the data
df = pd.read_csv('test_data/posts-dataset.csv')
df_selected = df[['author_id', 'text']]

concatenated_texts_by_author = df_selected.groupby('author_id')
num_authors = df_selected['author_id'].nunique()
print(num_authors)
print(f"Number all of authors: {num_authors}")
# print(f"Number of authors: {author}")
# Concatenating texts for each author
concatenated_texts_by_author = df_selected.groupby('author_id')['text'].apply(' '.join)
# Analyzing word frequency for each author
word_frequency_data = []
for author_id, text in concatenated_texts_by_author.items():
    
    tokens = re.findall(r'\b\w+\b', text, flags=re.UNICODE)
    word_counts = Counter(tokens)
    for word, count in word_counts.items():
        word_frequency_data.append((author_id, word, count))

# Creating DataFrame
word_frequency_df = pd.DataFrame(word_frequency_data, columns=['document', 'element', 'frequency_in_document'])

# Saving the data to a CSV file
word_frequency_df.to_csv('test_data/post_frequency.csv', index=False, mode='w', header=True)

# Printing the number of rows created
num_rows = len(word_frequency_df)
print("CSV file 'post_frequency.csv' with the frequency of elements from all authors has been created successfully.")
print(f"Number of rows in the file: {num_rows}")
