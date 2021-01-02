from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
from collections import defaultdict
from multiprocessing import Process
import json
import os 
import re
import heapq
import math

pattern = re.compile('[^a-zA-Z0-9]')

"""
Finds relative path to all files in DEV directory,
placing them in a list, where the document ID = index
"""
def find_files() -> list:
    files = list()
    
    for directory in [d for d in os.listdir("../DEV")]: # for each directory in the /DEV directory
        files += [os.path.join("../DEV/" + directory + "/", file) for file in os.listdir("../DEV/" + directory)] # construct the relative path to each individual file in each sub-directory
        
        
    if (os.path.exists("document_map.txt")): 
        print("File document_map.txt found, skipping document mapping to ID.")
    else:
        print("File document_map.txt not found, begin document mapping to ID.")
        
        with open("document_map.txt", "w") as f: # open every file we found in the last for loop 
            for id, file in enumerate(files):
                with open(file) as document:
                    f.write("{0} {1}\n".format(id, json.load(document)["url"])) # and assign a unique doc_id to each file
    
    
        print("Finished document mapping to ID.")
        
    return files

"""
Builds a part of the complete inverted index of our corpus as a dictionary
--> token : (document ID, frequency of token in that document)
"""
def build_partial_index(files, multithread, offset = -1):
    partial_index = defaultdict(list)
    
    if (multithread): # each thread tokenizes up to 10000 documents (in terms of doc_id) and offloads it
        const = 10000
        count = min((offset * const) + const, len(files)) # for thread #6, there's less than 10000 documents left so min() is used here
        i = offset * const
    elif (not multithread):
        i = 0
        count = 10
        offset = 0
        
    while (i < count):
        with open(files[i]) as f: # open document located at doc_id i
            #print(i, end = " ")
            current_file = tokenize(json.load(f)) # tokenize that document
            
            if (current_file is not None): # duplicate link detected (tokenize returns None if duplicate is detected)                
                tokens = current_file[1]
                important_set = current_file[2]
                doc_index = defaultdict(tuple) # we want to keep track of what terms this document has to normalize their log frequency weighting
                discovered = set() # we want to only look at unique tokens, because finding the count of token in list once is enough
                normalization = 0
                
                for token in tokens:
                    if (token != "" and token not in discovered):
                        if (token in important_set):
                            doc_index[token] = (str(i) + "!", 1 + math.log(tokens.count(token))) # if token is important, we add a ! at the end of the doc_id
                            normalization += (1 + math.log(tokens.count(token))) ** 2
                        else:
                            doc_index[token] = (i, 1 + math.log(tokens.count(token) / 2)) # otherwise, token is not important and we reduce its weighting
                            normalization += (1 + math.log(tokens.count(token) / 2)) ** 2
                            
                        discovered.add(token) # keep track of tokens we already dealt with in this document
        
                normalization = math.sqrt(normalization) # calculate the normalization factor
                
                for token, posting in doc_index.items(): 
                    partial_index[token].append((posting[0], round(posting[1] / normalization, 4))) # normalize each term with respect to the current document and add it to the partial_index
            
        if (not multithread and i % 10000 == 0 and i != 0): # if we do not use threads, we offload every 10000 documents and clear the dictionary
            offload_dict(partial_index, offset)
            offset += 1
        
        i += 1
        
    offload_dict(partial_index, offset)  # offloads automatically at the end of each thread/leftover files if not multi-threaded
    
    
"""
Returns a 3-tuple of the URL and its list of tokens
--> tuple[0] = URL
--> tuple[1] = list of tokens found in that URL
--> tuple[2] = important words
"""
def tokenize(json_data) -> (str, list):
    total_words = list()
    important_words = set()
    stemmer = PorterStemmer()
    
    if ("/#content" in json_data["url"]): # duplicate detection, URLs with #content are exact duplicates
        return None
    
    soup = BeautifulSoup(json_data["content"].encode(json_data["encoding"]).decode(json_data["encoding"]), "html.parser") # encode with given encoding and then decode to get exact text
    #print("[{0}]".format(json_data["url"]))
    
    # first loop handles important tokens located in specific HTML tags, tokenization logic: if a token contains a non-alphanumeric character, discard the whole token
    for line in (token.text.strip().lower() for token in soup.find_all("h1") + soup.find_all("h2") + soup.find_all("h3") + soup.find_all("b") + 
               soup.find_all("strong") + soup.find_all("title")): 
        for token in (tok for tok in line.lower().split() if re.sub(pattern, ' ', tok).isalnum()):
            token = stemmer.stem(token)
            if (len(token) >= 3):
                important_words.add(token)

    # second loop handles all text in the document, tokenization logic: if a token contains a non-alphanumeric character, discard the whole token
    for line in (line.strip() for line in soup.get_text().splitlines()):
        if (len(line) != 0):           
            for token in (tok for tok in line.lower().split() if re.sub(pattern, ' ', tok).isalnum()):
                token = stemmer.stem(token)
                if (len(token) >= 3):
                    total_words.append(token)
    
    return (json_data["url"], total_words, important_words)

"""
Writes current index to disk alphabetically for later usage,
emptying out current partial index
"""
def offload_dict(index_dict, count):
    with open("output{0}.txt".format(count), "a", encoding="utf-8") as f:
        for key, value in sorted(index_dict.items()): # offload memory to file as (token) | (doc_id)#(frequency of term) format
            f.write("{0} | ".format(key))
            
            for doc_freq in value:
                f.write("{0}#{1} ".format(doc_freq[0], doc_freq[1])) 
                
            f.write("\n")
            
    index_dict.clear() # clear memory

"""
Reads from disk and rebuilds the inverted index into
a dictionary
"""
def rebuild_dict(dict_list):
    index_dict = defaultdict(list)
    
    for token in dict_list:
        token = token.split("|")
        key = token[0].strip()
        
        for value in token[1].split():
            value = value.split("#")
            index_dict[key].append((value[0], value[1]))
        
    return index_dict

"""
Reconstructing partial indexes into one complete index
by reading from indices on disk and merging them
--> unfinished_index.txt in custom dictionary format with duplicate keys
"""
def reconstruct_whole_index():
    indicies = list()
    for num in range(6): # open each partial_index file
        f = open("output{0}.txt".format(num), "r")
        indicies.append(f)
            
    with open("unfinished_index.txt", "w") as f:
        f.writelines(heapq.merge(*indicies)) # function does not pull full index into memory, takes sorted streams/iterables as arguments and merges them into one file
        
    for f in indicies: # close each file stream
        f.close()
        
"""
Reads reconstructed index line by line, concatenating
duplicate key values and outputting it into a finished inverted index,
returning a count of the unique tokens in the index
"""
def clean_whole_index() -> int:  
    with open("unfinished_index.txt", "r") as index_in, open("index.txt", "w") as index_out:
        current_line = index_in.readline()
        current_key, current_values = current_line.split(" | ")
        tokens = 1;
        
        while current_line:
            next_line = index_in.readline()
            
            if (len(next_line) == 0): # end of index contains an empty line
                break
            
            next_key, next_values = next_line.split(" | ")
            
            if (next_key != current_key): # output line to index, as we are finished concatenating postings of the same token
                index_out.write("{0} | {1}".format(current_key, current_values))
                current_key = next_key
                current_values = next_values
                #print("{0} [{1}]".format(current_key, tokens))
                tokens += 1        
                #print(current_key)
            elif (next_key == current_key): # concatenate list of postings as the keys (tokens) are the same
                current_values = current_values.strip() + " " + next_values
    
            current_line = next_line
            
    with open("unfinished_index.txt", "w"): pass # delete unneeded file
    os.remove("unfinished_index.txt")
    
    return tokens - 1
"""
Find the pointer positions of every alphanumeric character
and store them onto disk for later usage (f.seek() to speed up queries)
"""
def build_seek_index():
    seek_string = "0123456789abcdefghijklmnopqrstuvwxyz"
    curr_ptr = 0
    
    with open("index.txt", "r") as index_in, open("seek_index.txt", "w") as index_out:
        line = index_in.readline()
        while (line and seek_string != ""):
            key = line.split(" | ")[0]
            if (key[0] == seek_string[0]): # if first letter of term is equal to an alphanumeric character we haven't recorded yet
                index_out.write("{0} {1}\n".format(seek_string[0], curr_ptr)) # we save its pointer location
                seek_string = seek_string[1:] # and remove that character from our wanted characters
            
            curr_ptr = index_in.tell() # keep track of previous pointer as readline() moves pointer to end of line
            line = index_in.readline()

if __name__ == "__main__":
    file_list = find_files()
    multithread = True
    
    # set all these to False to build index from scratch
    partial_indexes_built = True # if all partial indexes exist, set this to True
    inverted_index_built = True # if index.txt exists, set this to True
    seek_index_built = True # if seek_index.txt exists, set this to True
    
    if (multithread and not partial_indexes_built):
        print("Executing multi-process partial index building.")
        procs = list()
    
        for num in range(6):
            process = Process(target = build_partial_index, args = (file_list, multithread, num, ))
            process.start()
            procs.append(process)
    
        for process in procs:
            process.join()
            
        partial_indexes_built = True  
            
    elif (not multithread and not partial_indexes_built):
        print("Executing single-process partial index building.")
        build_partial_index(file_list, multithread)
        partial_indexes_built = True
    
    if (partial_indexes_built and not inverted_index_built):
        print("Reconstructing index from partial indexes found on disk.")
        
        reconstruct_whole_index()
        clean_whole_index()
        inverted_index_built = True
        
    if (not seek_index_built):
        print("Building seek_index.txt")
        build_seek_index()
        seek_index = True
    
    elif (seek_index):
        print("Files index.txt and seek_index.txt found; finished building index.")    
        