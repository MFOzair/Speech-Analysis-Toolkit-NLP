import os
import gensim
from president_helper import (
    read_file, 
    process_speeches, 
    merge_speeches, 
    get_president_sentences, 
    get_presidents_sentences, 
    most_frequent_words
)

# Step 1: Load all speech files
# Get a sorted list of all text files in the current directory
files = sorted([file for file in os.listdir() if file.endswith('.txt')])

# Ensure there are speech files to process
if not files:
    raise FileNotFoundError("No speech files found in the current directory.")

# Step 2: Read each speech file
# Read the contents of all speech files
speeches = [read_file(file) for file in files]

# Step 3: Preprocess speeches
# Process the text of each speech (e.g., tokenize, remove stop words)
processed_speeches = process_speeches(speeches)

# Step 4: Merge all speeches into sentences
# Combine all processed speeches into a single list of sentences
all_sentences = merge_speeches(processed_speeches)

# Step 5: Analyze all speeches
# Find the most frequently used words across all speeches
most_freq_words = most_frequent_words(all_sentences)

# Train a Word2Vec model for all speeches
all_prez_embeddings = gensim.models.Word2Vec(
    sentences=all_sentences, 
    vector_size=96, 
    window=5, 
    min_count=1, 
    workers=2, 
    sg=1  # Use skip-gram model
)

# Find words similar to "freedom" in the entire corpus
similar_to_freedom = all_prez_embeddings.wv.most_similar('freedom', topn=20)
print("\nWords similar to 'freedom' (All Presidents):")
print(similar_to_freedom)

# Step 6: Analyze a specific president (e.g., Franklin D. Roosevelt)
# Get sentences for President Franklin D. Roosevelt
roosevelt_sentences = get_president_sentences("franklin-d-roosevelt")

# Find the most frequently used words in Roosevelt's speeches
roosevelt_most_freq_words = most_frequent_words(roosevelt_sentences)

# Train a Word2Vec model for Roosevelt's speeches
roosevelt_embeddings = gensim.models.Word2Vec(
    sentences=roosevelt_sentences, 
    vector_size=96, 
    window=5, 
    min_count=1, 
    workers=2, 
    sg=1
)

# Find words similar to "freedom" in Roosevelt's speeches
similar_to_freedom_roosevelt = roosevelt_embeddings.wv.most_similar('freedom', topn=20)
print("\nWords similar to 'freedom' (Franklin D. Roosevelt):")
print(similar_to_freedom_roosevelt)

# Step 7: Analyze multiple presidents (e.g., Mount Rushmore presidents)
# Get sentences for Washington, Jefferson, Lincoln, and Theodore Roosevelt
rushmore_prez_sentences = get_presidents_sentences(
    ["washington", "jefferson", "lincoln", "theodore-roosevelt"]
)

# Find the most frequently used words in these presidents' speeches
rushmore_most_freq_words = most_frequent_words(rushmore_prez_sentences)

# Train a Word2Vec model for these presidents
rushmore_embeddings = gensim.models.Word2Vec(
    sentences=rushmore_prez_sentences, 
    vector_size=96, 
    window=5, 
    min_count=1, 
    workers=2, 
    sg=1
)

# Find words similar to "freedom" in the Rushmore presidents' speeches
rushmore_similar_to_freedom = rushmore_embeddings.wv.most_similar('freedom', topn=20)
print("\nWords similar to 'freedom' (Mount Rushmore Presidents):")
print(rushmore_similar_to_freedom)
