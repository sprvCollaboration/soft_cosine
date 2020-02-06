This repository was created to test the ideas presented in the:
Soft Similarity and Soft Cosine Measure:Similarity of Features in Vector Space Model paper.
The full publication can be found here:
http://www.scielo.org.mx/pdf/cys/v18n3/v18n3a7.pdf

The Soft_Cosine_class.py is the module that contains the 'code'
that implements the Soft Cosine similarity Math.

The Soft_Cosine_test.py script implements an example and demonstrates
the differences (at a high level) between Soft Cosine and plain vanilla cosine similarity.

To run the Soft_Cosine_test.py Program please follow the steps outlined below:
 1.Clone the Remote repo to a local directory
 2.Update the file path at the top of the Soft_Cosine_test.py file
 3.Ensure that the GoogleNews-vectors-negative300.bin is in your local directory as the code will need this to load the word embeddings that are 
   being used to compute the pairiwse similarity features.Due to the size of the file I dit not commit it to the repo. Please download by visiting this
   url: https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz 
