def ngram(sentence,n):
    input_list = [elem for elem in sentence.split(" ") if elem != '']
    return zip(*[input_list[i:] for i in xrange(n)])

#compare two documents for similarity
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
        


if __name__ == '__main__':

    # doc_one = "Hi, my name is Eric.  I work Syncano and deeply believe all people have the right to freedom.  But I'm very lonely sometimes.  It's usually around 12am when I haven't slept."
    # doc_two = "Hi, my name is Eric.  I work Syncano and deeply believe all people have the right to freedom."
    # print similarity_analysis(doc_one,doc_two)
    # print ngram("Hi, my name is Eric.  I work Syncano and deeply believe all people have the right to freedom.",1)

    # print ngram("Hi, my name is Eric.  I work Syncano and deeply believe all people have the right to freedom.",2)

    # print ngram("Hi, my name is Eric.  I work Syncano and deeply believe all people have the right to freedom.",3)

