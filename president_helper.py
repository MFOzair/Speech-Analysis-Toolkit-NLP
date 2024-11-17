import os
from nltk.tokenize import PunktSentenceTokenizer
from collections import Counter

# Function to read the content of a file
def read_file(file_name):
    """
    Reads the content of a file and returns it as a string.
    
    Args:
        file_name (str): The name of the file to read.
        
    Returns:
        str: The content of the file.
    """
    with open(file_name, 'r+', encoding='utf-8') as file:
        file_text = file.read()
    return file_text

# Function to process and tokenize speeches into words
def process_speeches(speeches):
    """
    Tokenizes speeches into word-tokenized sentences.
    
    Args:
        speeches (list): List of speech texts.
        
    Returns:
        list: Nested list of tokenized words for each speech.
    """
    word_tokenized_speeches = []
    sentence_tokenizer = PunktSentenceTokenizer()
    
    for speech in speeches:
        # Tokenize the speech into sentences
        sentence_tokenized_speech = sentence_tokenizer.tokenize(speech)
        word_tokenized_sentences = []
        
        for sentence in sentence_tokenized_speech:
            # Tokenize the sentence into words and clean punctuation
            word_tokenized_sentence = [
                word.lower().strip('.?!') 
                for word in sentence.replace(",", "").replace("-", " ").replace(":", "").split()
            ]
            word_tokenized_sentences.append(word_tokenized_sentence)
        
        word_tokenized_speeches.append(word_tokenized_sentences)
    return word_tokenized_speeches

# Function to merge all sentences from speeches into a single list
def merge_speeches(speeches):
    """
    Merges sentences from multiple speeches into a single list.
    
    Args:
        speeches (list): Nested list of tokenized words for each speech.
        
    Returns:
        list: Flattened list of all sentences.
    """
    all_sentences = []
    for speech in speeches:
        all_sentences.extend(speech)
    return all_sentences

# Function to get sentences from speeches of a specific president
def get_president_sentences(president):
    """
    Retrieves and processes sentences from speeches of a specific president.
    
    Args:
        president (str): Name of the president.
        
    Returns:
        list: List of tokenized sentences for the president.
    """
    files = sorted([file for file in os.listdir() if president.lower() in file.lower()])
    speeches = [read_file(file) for file in files]
    processed_speeches = process_speeches(speeches)
    return merge_speeches(processed_speeches)

# Function to get sentences from speeches of multiple presidents
def get_presidents_sentences(presidents):
    """
    Retrieves and processes sentences from speeches of multiple presidents.
    
    Args:
        presidents (list): List of president names.
        
    Returns:
        list: List of tokenized sentences for the presidents.
    """
    all_sentences = []
    for president in presidents:
        files = sorted([file for file in os.listdir() if president.lower() in file.lower()])
        speeches = [read_file(file) for file in files]
        processed_speeches = process_speeches(speeches)
        all_sentences.extend(merge_speeches(processed_speeches))
    return all_sentences

# Function to find the most frequent words in a list of sentences
def most_frequent_words(list_of_sentences):
    """
    Finds the most frequently occurring words in a list of sentences.
    
    Args:
        list_of_sentences (list): List of tokenized sentences.
        
    Returns:
        list: List of tuples with words and their frequencies.
    """
    all_words = [word for sentence in list_of_sentences for word in sentence]
    return Counter(all_words).most_common()
