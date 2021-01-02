from nltk.stem.porter import PorterStemmer
from collections import defaultdict
from itertools import islice
from urllib.parse import urldefrag
import math
import time
import re

seek_dict = defaultdict(str)

"""
Loads seek_index.txt into memory as a dictionary of
alphabetical positions of our index and its pointer position
"""
def load_seek_dict():
    global seek_dict
    
    with open("seek_index.txt") as f:
        for line in f:
            seek_dict[line[0]] = int(line[2:].strip())

"""
Calculates the dot product of 2 vectors
"""
def cosine(q_vector, d_vector):
    dot_prod = 0
    
    for i in range(len(q_vector)):
        dot_prod += q_vector[i] * d_vector[i]
        
    return dot_prod

"""
Algorithm:
    1) Parse query terms and stem them
    2) Use alphabetical seek positions of index.txt to find terms quickly
    3) Calculate query ltc and normalize the weight
    4) Calculate the cosine similarity of query ltc and document lnc
Returns a dictionary of documents and their cosine similarity score with the query
"""
def query(query_string) -> defaultdict(list):
    stemmer = PorterStemmer()
    query_terms = [stemmer.stem(query) for query in re.sub(r'[^a-zA-Z0-9]', ' ', query_string.lower()).split() if len(stemmer.stem(query)) > 2] # stem the query and make sure its length is > 2
    relevance = defaultdict(list)
    query_vector = list()
    normalization = 0
    
    with open("index.txt", "r") as f:
        for term in query_terms:
            f.seek(seek_dict[term[0]]) # find file position where query is close by starting at the same alphanumeric position in the index
            line = f.readline()
            
            while (line.split(" | ")[0] != term):
                if (line.split(" | ")[0] > term): # if term found in index > query_term, there is no point iterating further 
                    return relevance              # and we will not find the term in the index; continue onto next query_term
                line = f.readline()
                

                
            parsed_line = line.split(" | ")
            term_idf = (1 + math.log(query_terms.count(term))) * math.log(55393/len(parsed_line[1].split())) # calculate each query term's idf
            query_vector.append(term_idf) # vector representation of query with idf (not normalized yet)
            normalization += term_idf ** 2 # keeping track of normalization factor to correctly normalize query vector weight
            
            for posting in parsed_line[1].split():
                doc, score = posting.split("#")
                    
                if ("!" in doc):
                    doc = doc[:-1]
                    
                relevance[int(doc)].append(float(score)) # dictionary of all relevant documents containing at least a part of the query term, where key = term and value = normalized lnc

    # return a dictionary comprehension where we find the cosine similarity of ltc (normalized query vector) 
    # and lnc (normalized document vector) only if they have equal dimensions
    # key = doc_id, value = lnc.ltc ranking based on cosine similarity
    return {key : cosine([dimension / math.sqrt(normalization) for dimension in query_vector], value) for key, value in relevance.items() if len(value) == len(query_vector)} 

"""
Outputs the results of a query, ordered by cosine similarity
"""
def print_urls(query_results, display = 10):
    unique_urls = set()
    for i, (doc, score) in enumerate(sorted(query_results.items(), key=lambda x:x[1], reverse=True)):
        with open("document_map.txt") as f:
            line = next(islice(f, doc, doc + 1)).split(" ")[1].strip() # go to exact line in document_map.txt to find URL
            if (urldefrag(line)[0] not in unique_urls): # duplicate detection, URL might not be unique after we defrag it so we do not output non-unique URLs
                unique_urls.add(urldefrag(line)[0])
                print("{}".format(urldefrag(line)[0]))
                
        if (i >= display - 1): # display only top 10 results or custom amount of results
            break
    
if __name__ == "__main__":
    load_seek_dict()

    while (True):
        q = input("Enter a search query (or enter !q to quit): ")
        q = q.lower().strip()
        
        if (q == "!q"):
            break
        
        start = time.time()
        results = query(q)
        end = time.time()
        print("Found {} results for query '{}' in {:.3f} ms\n".format(len(results), q, (end - start) * 1000)) 
        
        if (len(results) == 0):
            continue  
        else:
            display = input("How many results would you like to see? (press Enter to see default = 10 results): ")
        
        if (display == ""):
            print()
            print_urls(results)
        else:
            try:
                display = float(display)
                print_urls(results, display)
            except ValueError:
                print("Incorrect format detected; showing default = up to 10 results.\n")
                print_urls(results)
        
        print()
                
    
        