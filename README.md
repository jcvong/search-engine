# Search Engine\

A search engine centered around finding relevant articles given a downloaded corpus.\

## Indexer (must be run before retriever)\
indexer.py contains all the functions required to build an effective index, complete with partial index offloading and reconstruction. Index construction can be split into 3 parts: partial index building, inverted index building, and seek index building. Options also include a single and multi-threaded approach, resulting in the same output: *6 partial index files, a complete index.txt file, a seek_index.txt file, and a document_map.txt file* (used for docID -> URL mapping).

The inverted index is built in a dictionary representation, where keys are terms in the corpus and values are posting lists consisting of the document Id and its normalized tf-idf weighting (also known as **lnc**)

**NOTE:** indexer.py assumes the user doesn't have any of the prequisite files (partial index text files, document_map, seek_index, index) on disk and will start a multi-threaded partial index construction as the default action when ran. This can be changed inside the main method.


