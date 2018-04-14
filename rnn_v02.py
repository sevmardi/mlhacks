import csv
import nltk
import itertools

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

print("reading the csv file...")
with open('data/reddit-comments', 'rb') as f:
    reader = csv.reader(f, skipinitalspace=True)
    reader.next()
    # split full comments into sentances
    sentances = itertools.chain(
        *[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    # Append SENTANCE_START and SENTANCE_END
    sentances = ["%s %s %s" % (
            sentence_start_token, x, sentence_end_token) for x in sentences]
print("Parsed %d sentances." % (len(sentances)))

# tokenize the sentances into words
tokenized_sentances = [nltk.word_tokenize(sent) for sent in sentances]

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentances))
print("Found %d unique words tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
