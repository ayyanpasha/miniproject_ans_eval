from flask import Flask
 
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)
 
# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    DATASET_CSV ='Define OOPs.csv'

    import pandas as pd

    oop_dataset=pd.read_csv(DATASET_CSV, encoding="ISO-8859-1")
    oop_dataset.head()
    oop_dataset=oop_dataset[:1000]

    df = pd.DataFrame(oop_dataset, columns = ['Question No.', 'Question', 'USN', 'Answer Key', 'Student Answer','Marks'])


    DATASET ='OOP_Dataset_MSRIT.csv'

#@title Spacy
    import spacy.cli
    nlp = spacy.load("en_core_web_lg")

    import numpy as np
    import itertools
    from sklearn.cluster import KMeans
    import pprint
    import nltk
    from nltk.tokenize import word_tokenize
    oop_dataset=pd.read_csv(DATASET)
    oop_dataset.head()
    oop_dataset=oop_dataset[:1000]

    df = pd.DataFrame(oop_dataset, columns = ['Student Answer'])
    stud_ans=df['Student Answer'].tolist()

    df = pd.DataFrame(oop_dataset, columns = ['Answer Key'])
    key=df['Answer Key'].tolist()
    key=key[0]
    ans_doc=nlp(key)

    df = pd.DataFrame(oop_dataset, columns = ['USN'])
    USN=df['USN'].tolist()

#@title List of Keywords

    def extract_POS(sample_doc):
        res=[]
        for chk in sample_doc.noun_chunks:
            tmp=""
            for tkn in chk:
                if (tkn.pos_ in ['NOUN','PROPN','ADJ'] ):
                    if (not(tkn.is_stop) and not(tkn.is_punct)):
                        tmp = tmp + tkn.text.lower() + " "
            if(tmp.strip()!=""):
                res.append(tmp.strip())
        return list(dict.fromkeys(res))

    key_POS=extract_POS(ans_doc)



    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    doc_embedding = model.encode([key])
    candidate_embeddings = model.encode(key_POS)

#@title Cosine Similarity

    from sklearn.metrics.pairwise import cosine_similarity

    top_n = 7
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [key_POS[index] for index in distances.argsort()[0][-top_n:]]


#@title Matching keywords

    def matching_keywords(stdlst,keylst):
        #matched list
        res=[]
        #unmatched list
        tmpres=[]
        for x in stdlst:
            if (x in keylst):
                res.append(x)
        return res



#@title Assign Weights to Keywords

    def dictionary_with_weights(words):

# nouns are given weightage as 10
# proper nouns are given weightage as 7
# adjectives are given weightage as 5

        categorized_words = {'10': [], '7': [], '5': []}
        for word in words:
            doc = nlp(word)
            pos = doc[0].pos_
            if pos in ['NOUN', 'PRON']:
                categorized_words['10'].append(word)
            elif pos == 'PROPN':
                categorized_words['7'].append(word)
            elif pos == 'ADJ':
                categorized_words['5'].append(word)
        return categorized_words

    keywords_scores = dictionary_with_weights(keywords)

#@title Final Scoring

    total=0
    count=0
    for i in keywords_scores['10']:
        count+=1
    total=total+count*10

    count=0
    for i in keywords_scores['7']:
        count+=1
    total=total+count*7

    count=0
    for i in keywords_scores['5']:
        count+=1
    total=total+count*5


    count=1
    for stud_doc in stud_ans:
        if(type(stud_doc)==float):
            stud_doc=str(stud_doc)
        stud_doc=nlp(stud_doc) 
        std_POS=extract_POS(stud_doc)
#then apply match function
        matched=matching_keywords(std_POS,keywords)
        stud_score=0
#percentage of matched keywords along with weights
        for i in matched:
            if i in keywords_scores['10']:
                stud_score=stud_score+10
            elif i in keywords_scores['7']:
                stud_score=stud_score+7
            elif i in keywords_scores['5']:
                stud_score=stud_score+5

        print("Evaluation of student:",count,"'s response:")
        print("Matching percentage with keywords:",len(matched)/len(keywords))
        print("Relative score using weights:",stud_score/total)

        print("Matched Special POS keywords:", matched)
        print("\n")
        count+=1
    return "Hello"
 
# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(debug=False,host='0.0.0.0')
