#An Introduction to Natural Language Processing

Natural language processing (NLP) is the study of translation of human language into something a computer can understand and manipulate.  The areas of study within NLP are diverse and require a somewhat disparate set of skills.  Many of these have evolved with the prevalence of machine learning techniques and practices.

In this lecture we will focus on text based machine learning techniques and learn how to make use of these techniques to do text classification and analysis.

##N-grams

A fundamental technique in natural language processing is the ability to create N-grams.  The best way to understand an N-gram, is to see it.  

We will use the same sentence through out:

Hi, my name is Eric.  I work Syncano and deeply believe all people have the right to freedom.

A 1-gram of that sentence would be:


[('Hi,'), ('my'), ('name'), ('is'), ('Eric.'), ('I'), ('work'), ('Syncano'), ('and'), ('deeply'), ('believe'), ('all'), ('people'), ('have'), ('the'), ('right'), ('to'), ('freedom.')]


A 2-gram of that sentence would be:

[('Hi,', 'my'), ('my', 'name'), ('name', 'is'), ('is', 'Eric.'), ('Eric.', 'I'), ('I', 'work'), ('work', 'Syncano'), ('Syncano', 'and'), ('and', 'deeply'), ('deeply', 'believe'), ('believe', 'all'), ('all', 'people'), ('people', 'have'), ('have', 'the'), ('the', 'right'), ('right', 'to'), ('to', 'freedom.')]

A 3-gram of that sentence would be:

[('Hi,', 'my', 'name'), ('my', 'name', 'is'), ('name', 'is', 'Eric.'), ('is', 'Eric.', 'I'), ('Eric.', 'I', 'work'), ('I', 'work', 'Syncano'), ('work', 'Syncano', 'and'), ('Syncano', 'and', 'deeply'), ('and', 'deeply', 'believe'), ('deeply', 'believe', 'all'), ('believe', 'all', 'people'), ('all', 'people', 'have'), ('people', 'have', 'the'), ('have', 'the', 'right'), ('the', 'right', 'to'), ('right', 'to', 'freedom.')]

As you can see the number in front of gram determines how many elements each split we have.  Also, we only increment the element by one in the sentence.  This creates small phrases, which make up the elements of the n-gram.

Here's the code I used to generate the above sequences:

```
def ngram(sentence,n):
    input_list = [elem for elem in sentence.split(" ") if elem != '']
    return zip(*[input_list[i:] for i in xrange(n)])
```

The zip function will zip n many lists together into one list, where the elements of each list will become tuples. A small piece of syntactic sugar you may not be familiar with is the * in front of the list comprehension.  

To understand the difference here let's look at an example:

```
#basic example
def thing(*x): print x
>>> thing([[elem] for elem in xrange(5)])
([[0], [1], [2], [3], [4]],)
>>> thing(*[[elem] for elem in xrange(5)])
([0], [1], [2], [3], [4])

#with zip
>>> zip([[elem] for elem in xrange(5)])
[([0],), ([1],), ([2],), ([3],), ([4],)]
>>> zip(*[[elem] for elem in xrange(5)])
[(0, 1, 2, 3, 4)]
```

As you can see, the * being passed into a function simply empties the elements of the list comprehension one by one, which is exactly what we want for our zip function.

So what can you do with your n-gram?  

N-grams have a number of uses [wikipedia](http://en.wikipedia.org/wiki/N-gram) illustrates some of these.

However, the simplest (and one of the best) use case is for comparison of writing samples to determine authorship.  The notion is, an author will continually the same style throughout his or her writing, therefore you can expect phrasing to be repeated across different documents.  Of course, word choice will likely be similar (assuming the author is talking about a similar subject) and the phrases should be similar as well. 

So using n-grams we can determine, approximate measures (albeit very simple measures) for when two pieces of writing are similar.  

So how might we do that:

```
def similarity_analysis(doc_one,doc_two):
    ngrams_one = [ngram(doc_one,elem) for elem in xrange(1,4)]
    ngrams_two = [ngram(doc_two,elem) for elem in xrange(1,4)]
    
    #longer body of text should be looped through
    if len(ngrams_one) < len(ngrams_two):
        ngrams_one,ngrams_two = ngrams_two, ngrams_one
    word_choice_count = 0 
    phrase_choice_count = 0
    for elem in ngrams_one[0]:
        if elem in ngrams_two[0]:
            word_choice_count += 1
    word_choice_similarity = float(word_choice_count)/len(ngrams_one[0])
    
    phrases_one = ngrams_one[1] + ngrams_one[2]
    phrases_two = ngrams_two[1] + ngrams_two[2]
    for elem in phrases_one:
        if elem in phrases_two:
            phrase_choice_count += 1
    phrase_choice_similarity = float(phrase_choice_count)/len(phrases_one)
    return word_choice_similarity, phrase_choice_similarity
```

Thus we have meta information about our writing, that is extremely simple.  But potentially power.  

##Text classification

Run through these:

http://textblob.readthedocs.org/en/latest/_modules/textblob/classifiers.html


##Document similarity

Document similarity:

tf-idf: http://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity

http://www.academia.edu/188660/Analysing_Document_Similarity_Measures

Cosine similarity:

http://en.wikipedia.org/wiki/Cosine_similarity

http://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

http://en.wikipedia.org/wiki/Hamming_distance

http://en.wikipedia.org/wiki/Jaccard_index

http://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance

http://en.wikipedia.org/wiki/Levenshtein_distance

References:

[intro to ngrams in haskell](http://nlpwp.org/book/chap-ngrams.xhtml) 
