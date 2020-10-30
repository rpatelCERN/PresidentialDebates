import scattertext as st
from scattertext import SampleCorpora, RankDifference, dense_rank, PyTextRankPhrases, AssociationCompactor, \
    produce_scattertext_explorer
from scattertext import CorpusFromParsedDocuments
import spacy
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from spacy.matcher import PhraseMatcher,Matcher
import pytextrank
from pprint import pprint
import pandas as pd
import numpy as np
import glob, os
import sklearn
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


matplotlib.rc('font',family='monospace')
plt.style.use('ggplot')
fig, ax = plt.subplots()
plt.xlabel("Debate Topics")
plt.ylabel("Count Freq.")


def BuildDF(partydict,DoAll=False,ADebate=""):
    #
    if DoAll:Debate_df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "RawCSV/*.csv" ))),ignore_index=True)
    else:Debate_df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', "RawCSV/*%s*.csv" %ADebate))),ignore_index=True)
    #Debate_df=Debate_df.head(100)
    Debate_df=Debate_df[~(Debate_df.Response.isnull())]

    Candidates=[key for key in partydict.keys()]
    Debate_df["Candidates"]=Debate_df["Speaker"].isin(Candidates)
    Debate_df = Debate_df.loc[:, ~Debate_df.columns.str.contains('^Unnamed')]
    CandidateDF=Debate_df[Debate_df["Candidates"]==True]
    party=[partydict[s] for s in CandidateDF["Speaker"].to_list()]
    ### Integer representaition
    PartyInt={"Democratic":0, "Republican":1}
    party=[PartyInt[p] for p in party]


    FascilatorDF=Debate_df[Debate_df["Candidates"]==False]
    CandidateDF.drop('Candidates',inplace=True,axis=1)
    FascilatorDF.drop('Candidates',inplace=True,axis=1)
    Debate_df.drop('Candidates',inplace=True,axis=1)
    CandidateDF.insert(CandidateDF.shape[1],"party",party,True)
    return CandidateDF,FascilatorDF,Debate_df
def CreateScatterText(CandidateDF,compact):
    nlp = spacy.load('en')
    CandidateDF=CandidateDF.sample(frac=1)#### Shuffle speakers for the meta data
    CandidateDF = CandidateDF.assign(
        parse=lambda df: df.Response.apply(nlp)
        )
    print(CandidateDF.head())
    corpus = CorpusFromParsedDocuments(
        CandidateDF,
        category_col='party',
        parsed_col='parse',
        feats_from_spacy_doc=PyTextRankPhrases()
    ).build().compact(AssociationCompactor(compact, use_non_text_features=True))
    print('Aggregate PyTextRank phrase scores')
    term_category_scores = corpus.get_metadata_freq_df('')
    print(term_category_scores)

    term_ranks = np.argsort(np.argsort(-term_category_scores, axis=0), axis=0) + 1

    metadata_descriptions = {
        term: '<br/>' + '<br/>'.join(
            '<b>%s</b> TextRank score rank: %s/%s' % (cat, term_ranks.loc[term, cat], corpus.get_num_metadata())
            for cat in corpus.get_categories())
        for term in corpus.get_metadata()
    }

    category_specific_prominence = term_category_scores.apply(
        lambda r: r.Democratic if r.Democratic > r.Republican else -r.Republican,
        axis=1
    )

    html = produce_scattertext_explorer(
        corpus,
        category='Democratic',
        not_category_name='Republican',
        minimum_term_frequency=1,
        pmi_threshold_coefficient=0,
        width_in_pixels=1500,
        transform=dense_rank,
        use_non_text_features=True,
        metadata=corpus.get_df()['Speaker'],
        scores=category_specific_prominence,
        #scores=term_category_scores,
        sort_by_dist=False,
        # ensure that we search for term in visualization
        topic_model_term_lists={term: [term] for term in corpus.get_metadata()},
        topic_model_preview_size=0,  # ensure singleton topics aren't shown
        metadata_descriptions=metadata_descriptions,
        use_full_doc=True
    )

    file_name = 'demo_pytextrank_prominence.html'
    open(file_name, 'wb').write(html.encode('utf-8'))
    print('Open %s in Chrome or Firefox.' % file_name)


    html = produce_scattertext_explorer(
        corpus,
        category='Democratic',
        not_category_name='Republican',
        width_in_pixels=1500,
        minimum_term_frequency=1,
        pmi_threshold_coefficient=0,
        transform=dense_rank,
        use_non_text_features=True,
        metadata=corpus.get_df()['Speaker'],
        term_scorer=RankDifference(),
        topic_model_term_lists={term: [term] for term in corpus.get_metadata()},
        topic_model_preview_size=0,  # ensure singleton topics aren't shown
        metadata_descriptions=metadata_descriptions,
        use_full_doc=True
    )

    file_name = 'demo_pytextrank_rankdiff.html'
    open(file_name, 'wb').write(html.encode('utf-8'))
    print('Open %s in Chrome or Firefox.' % file_name)

def display_topics(model, feature_names, no_top_words):
    topicWords=[]
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(", ".join([feature_names[i]for i in topic.argsort()[:-no_top_words - 1:-1]]))
        topicWords.append(", ".join([feature_names[i]for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return topicWords
#### Make this a new function
def InputTextFeatures(docs,threshold,stop_words):
    scikittext=[]
    #article=[]
    rank=[]
    frequency=[]
    phrase=[]
    for doc in docs:
        phrases=doc._.phrases
        #print(doc.text)
        for p in phrases:
            if any(stop in p.text for stop in stop_words) :continue
            rank.append(p.rank)
            frequency.append(p.count)
            phrase.append(p.text)
            #print(p.rank,p.text)
            #if(p.count<2):continue
            #if(p.rank>0.1):
            if p.rank>threshold:
                docClean=nlpPyRank(p.text)
                NameCheck=False
                tokenpos=[token.pos_ for token in docClean]
                if any(pos=="PROPN" for pos in tokenpos):continue
                scikittext.append(p.text)

    #mapofWords={'textrank':rank,'frequency':frequency,'phrase':phrase}
    #outputDF=pd.DataFrame(mapofWords)
    #outputDF.to_csv("TestPyTextRank.csv")
    return scikittext

def CreateTopics(scikittext,stop_words,topics=6,doLDA=False):
    #### Make new function
    count_vectorizer = CountVectorizer(ngram_range=(2,4),stop_words=stop_words)#countVectorizer(ngram_range=(2,4),stop_words=stop_words)##### Set MinDF and MaxDF
    count = count_vectorizer.fit_transform(scikittext)
    count_feature_names = count_vectorizer.get_feature_names()

    if doLDA:
        lda = LatentDirichletAllocation(n_components=5,random_state=0).fit(count)
        return lda,count_feature_names;

    else:
        nmf = NMF(n_components=topics, random_state=1,beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,l1_ratio=.5).fit(count)
        return nmf,count_feature_names;

partydict={"OBAMA":"Democratic","MCCAIN":"Republican","ROMNEY":"Republican","MR. KENNEDY":"Democratic",
    "MR. NIXON":"Republican","MR. FORD":"Republican","MR. CARTER":"Democratic","MR. REAGAN":"Republican",
    "MR. MONDALE":"Democratic","GEORGE W. BUSH":"Republican","GEORGE H.W. BUSH":"Republican","DUKAKIS":"Democratic",
    "BILL CLINTON":"Democratic","HILLARY CLINTON":"Democratic","DOLE":"Republican","GORE":"Democratic",
    "KERRY":"Democratic","TRUMP":"Republican","BIDEN":"Democratic"}

#partydict={"OBAMA":"Democratic","MCCAIN":"Republican","ROMNEY":"Republican","MR. KENNEDY":"Democratic","SENATOR KENNEDY":"Democratic",
    #"MR. NIXON":"Republican","MR. FORD":"Republican","MR. CARTER":"Democratic","MR. REAGAN":"Republican","MR. MONDALE":"Democratic",
    #"BUSH":"Republican","PRESIDENT BUSH":"Republican","DUKAKIS":"Democratic","CLINTON":"Democratic","DOLE":"Republican","GORE":"Democratic","KERRY":"Democratic","TRUMP":"Republican","BIDEN":"Democratic"}
speakers=[key.lower() for key in partydict.keys() ]
#CandidateDF,FascilatorDF,Debate_df=BuildDF(partydict,False,"Reagan-Mondale")
CandidateDF,FascilatorDF,Debate_df=BuildDF(partydict,True)
#CreateScatterText(CandidateDF,200000)
PromptsModeratorComments=FascilatorDF["Response"].to_list()
#PromptsModeratorComments=[p.lower() for p in PromptsModeratorComments]
nlpPyRank = spacy.load("en_core_web_lg")
stop_words=['evening','night','senator','vice','president','united','states','question','questions','seconds','second','minutes','minute','last','mr','time','governor','closing','opening','rebuttal','segment',"follow up","transcription","town hall"]
stop_words.extend(", ".join(speakers).split(" "))
Moderatorlist=list(set(FascilatorDF["Speaker"].to_list()))
stop_words.extend(", ".join(Moderatorlist).split())
for w in stop_words:nlpPyRank.vocab[w].is_stop = True;
tr = pytextrank.TextRank()
nlpPyRank.add_pipe(tr.PipelineComponent, name="textrank", last=True)
docs=nlpPyRank.pipe(PromptsModeratorComments)

#scikittext=InputTextFeatures(docs,0.1,stop_words)
#CreateTopics(scikittext,stop_words,6,False)
#topicWords=display_topics(nmf,count_feature_names,20)


##### At this point i have the key debate topics
##############


topic=0
CandResponses=CandidateDF["Response"].to_list()
#CandResponses=[cand.lower() for cand in CandidateDF["Response"].to_list()]
Speakers=[cand.lower() for cand in CandidateDF["Speaker"].to_list()]

Canddocs=nlpPyRank.pipe(CandResponses)

##### Make part of user definition functions
matcher = Matcher(nlpPyRank.vocab)
#### Race relations and racial sensitivity
RACE=[{"LOWER":"black"},{"POS":"NOUN"}]
matcher.add("Race",None,RACE)
matcher.add("Race",None,[{"LOWER":{"IN":["racist","racial","segregated","hispanic","discrimination"]}}])
matcher.add("Race",None,[{"LOWER":"civil"},{"LOWER":"rights"}])
matcher.add("Race",None,[{"LOWER":"white"},{"LOWER":"supremacists"}])
matcher.add("Race",None,[{"LOWER":"affirmative"},{"LOWER":"action"}])
matcher.add("Race",None,[{"LOWER":"race"},{"LOWER":"relations"}])
matcher.add("Race",None,[{"LOWER":"hate"},{"LEMMA":"crime"}])
matcher.add("Race",None,[{"LOWER":"non"},{"IS_PUNCT":True,"OP":"?"},{"LOWER":"whites"}])

#### Gun control:
matcher.add("Gun Control",None,[{"LOWER":"gun"},{"LOWER":"control", "OP":"?"}])
matcher.add("Gun Control",None,[{"LOWER":"assault"},{"LOWER":"weapons"}])
matcher.add("Gun Control",None,[{"LOWER":"high"},{"LOWER":"capacity"},{"LEMMA":"magazine"}])
#### Oil:
matcher.add("Oil Industry",None,[{"LOWER":"oil"},{"LOWER":{"IN":["imports","industry","price","embargos"]},"OP":"?"}])
matcher.add("Oil Industry",None,[{"LOWER":{"IN":["arab","foreign"]}},{"LOWER":"oil"}])
matcher.add("Oil Industry",None,[{"LOWER":"inequitable"},{"LOWER":"depletion"}])
matcher.add("Oil Industry",None,[{"LOWER":"depletion"},{"LOWER":"allowance"}])
matcher.add("Oil Industry",None,[{"LOWER":"gas"},{"LEMMA":"price"}])
matcher.add("Oil Industry",None,[{"LOWER":"fossil"},{"LEMMA":"power"}])

#### Federal Spending
matcher.add("Federal Spending",None,[{"LOWER":{"IN":["deficit","surplus"]}}])
matcher.add("Federal Spending",None,[{"LOWER":"federal"},{"LOWER":{"IN":["spending","debt","deficit"]}}])
matcher.add("Federal Spending",None,[{"LOWER":"spending"},{"LOWER":"freeze"}])
matcher.add("Federal Spending",None,[{"LOWER":"budgetary"},{"LOWER":"restraints"}])
matcher.add("Federal Spending",None,[{"LOWER":"discretionary"},{"LOWER":"spending"}])

#### Health care
matcher.add("Health Care",None,[{"LOWER":"health"},{"LOWER":{"IN":["care","insurance","policies"]}}])
matcher.add("Health Care",None,[{"LOWER":"private"},{"LOWER":"insurance"}])
matcher.add("Health Care",None,[{"LOWER":"public"},{"LOWER":"health"}])
matcher.add("Health Care",None,[{"LOWER":"socialized"},{"LOWER":"medicine"}])
matcher.add("Health Care",None,[{"LOWER":"insurance"},{"LOWER":"companies"}])
matcher.add("Health Care",None,[{"LOWER":"prescription"},{"LOWER":"drugs"}])

#### Economic Growth/Recovery
matcher.add("Economy", None,[{"LOWER":"economic"},{"LOWER":{"IN":["problems","growth","issues","situation","crisis","realities"]}}])
matcher.add("Economy", None,[{"LOWER":"financial"},{"LOWER":{"IN":["bailout","crisis","rescue","problems"]}}])
matcher.add("Economy",None,[{"LOWER":"small", "OP":"?"},{"LOWER":"businesses"}])
matcher.add("Economy",None,[{"LOWER":{"IN":["gdp","economy"]}}])
matcher.add("Economy",None,[{"LOWER":"bank"},{"LOWER":"crisis"}])
matcher.add("Economy",None,[{"LOWER":"stock"},{"LOWER":"market"}])
matcher.add("Economy",None,[{"LOWER":"inflation"},{"LOWER":"rate"}])
matcher.add("Economy",None,[{"LOWER":"savings"},{"LOWER":"accounts"}])
matcher.add("Economy",None,[{"LOWER":"underlying"},{"LOWER":"inflation"}])
matcher.add("Economy",None,[{"LOWER":{"IN":["good","paying","american","more"]}},{"LOWER":"jobs"}])
matcher.add("Economy",None,[{"LOWER":"full"},{"LOWER":"employment"}])
matcher.add("Economy",None,[{"LOWER":"unemployment"}])
matcher.add("Economy",None,[{"LOWER":"working"},{"LOWER":"families"}])
matcher.add("Economy",None,[{"LOWER":"wage"},{"LEMMA":"earner"}])

### Social Welfare programs: e.g. Social Security, housing subsidies, federal minimum wage,
matcher.add("Social Welfare",None,[{"LOWER":"social"},{"LOWER":"security"}])
matcher.add("Social Welfare",None,[{"LOWER":"housing"},{"LOWER":"subsidies"}])
matcher.add("Social Welfare",None,[{"LOWER":"minimum"},{"LOWER":"wage"}])
matcher.add("Social Welfare",None,[{"LOWER":"prevailing"},{"LOWER":"wages"}])
matcher.add("Social Welfare",None,[{"LOWER":"abnormal"},{"LOWER":"poverty"}])


#### Public education
matcher.add("Public Education",None,[{"LOWER":"public"},{"LOWER":"education"}])
matcher.add("Public Education",None,[{"LEMMA":"school"},{"LOWER":{"IN":["buildings","teachers","districts","violence"],"OP":"?"}}])
matcher.add("Public Education",None,[{"LOWER":"teacher"},{"LOWER":"salaries"}])
matcher.add("Public Education",None,[{"LOWER":"high"},{"LOWER":"school"},{"LOWER":"graduates"}])

#### Abortion
matcher.add("abortion",None,[{"LOWER":{"IN":["abortions","abortion"]}}])

#####National security/Defense
matcher.add("National Defense",None,[{"LOWER":"defense"},{"LOWER":{"IN":["spending","budget"]}}])
matcher.add("National Defense",None,[{"LOWER":"war"},{"LOWER":"materials"}])
matcher.add("National Defense",None,[{"LOWER":"military"},{"LOWER":{"IN":["forces","equipment","action","technology","power"]}}])
matcher.add("National Defense",None,[{"LOWER":"hostile"},{"LOWER":"actors"}])
matcher.add("National Defense",None,[{"LOWER":"terrorist"},{"LOWER":"attacks"}])
matcher.add("National Defense",None,[{"LOWER":"nuclear"},{"LOWER":"warheads"}])
matcher.add("National Defense",None,[{"LOWER":"national"},{"LOWER":"security"}])
matcher.add("National Defense",None,[{"LOWER":"more"},{"LOWER":"troops"}])
matcher.add("National Defense",None,[{"LOWER":"foreign"},{"LOWER":"hot"},{"LOWER":"spots"}])
matcher.add("National Defense",None,[{"LOWER":"back"},{"LOWER":"door"},{"LOWER":"draft"}])

#### Immigration
matcher.add("Immigration",None,[{"LOWER":"immigration"},{"LOWER":"reform"}])
matcher.add("Immigration",None,[{"LOWER":"green"},{"LOWER":"cards"}])
matcher.add("Immigration",None,[{"LOWER":"illegal"},{"LOWER":{"IN":["immigration","workers"]}}])

#### Climate change /Alternative energy
matcher.add("Climate Change",None,[{"LOWER":"air"},{"LOWER":"pollution"}])
matcher.add("Climate Change", None, [{"LOWER":"climate"}, {"LOWER":"change"}])
matcher.add("Climate Change", None, [{"LOWER":"global"}, {"LOWER":"energy"}])
matcher.add("Climate Change", None, [{"LOWER":"global"}, {"LOWER":"warming"}])
matcher.add("Climate Change", None, [{"LOWER":"alternative"}, {"LOWER":"energy"}])
matcher.add("Climate Change", None, [{"LOWER":"energy"}, {"LOWER":{"IN":["secretary", "policy"]}}])
matcher.add("Climate Change", None, [{"LOWER":"conservation"}, {"LOWER":"efforts"}])

#### Return these
dictionaryFoundKeywords={"Race":[],"Immigration":[], "Gun Control":[],"Climate Change":[], "Federal Spending":[], "abortion":[], "National Defense":[],
"Oil Industry":[], "Economy":[], "Public Education":[],"Health Care":[], "Social Welfare":[]}
dictionaryCounts={"Race":0,"Immigration":0, "Gun Control":0,"Climate Change":0, "Federal Spending":0, "abortion":0, "National Defense":0,
"Oil Industry":0, "Economy":0, "Public Education":0,"Health Care":0, "Social Welfare":0  }

##### This function will record how many responses match a token based matchpattern
TopicMatches=[]
for doc in Canddocs:### Loop in order of the response column
    matches=matcher(doc)
    if(len(matches)>0):TopicMatches.append(True)
    else:TopicMatches.append(False)
    for match_id, start, end in matches:
        Topic=nlpPyRank.vocab.strings[match_id]
        dictionaryCounts[Topic]=dictionaryCounts[Topic]+1
        dictionaryFoundKeywords[Topic].append(doc[start:end].text)
for key in dictionaryFoundKeywords.keys():dictionaryFoundKeywords[key]=list(set(dictionaryFoundKeywords[key]))
CandidateDF.insert(CandidateDF.shape[1],column="MatchToTopic",value=TopicMatches)
###Then require a topic match
Topics=[key for key in dictionaryCounts.keys()]
Counts=[dictionaryCounts[key] for key in dictionaryCounts.keys()]
Keywords=[", ".join(dictionaryFoundKeywords[key]) for key in dictionaryCounts.keys()]
Dfout=pd.DataFrame.from_dict({"Topic":Topics,"Count":Counts,"Keywords":Keywords})
Dfout.to_csv("OutputTotalDebateTopics.csv")
CandidateDF=CandidateDF[CandidateDF.MatchToTopic==True]
CandDocsKeywords=nlpPyRank.pipe(CandidateDF.Response.to_list())
MatchedWords=[]
Topics=[]

for doc in CandDocsKeywords:### Loop in order of the response column
    matches=matcher(doc)
    ListOfMatchedWords=[]
    ListOfTopics=[]
    #### need to limit doc size for BERT to up to 512 tokens
    if(len(doc)>512):print(doc.text, len(doc))#### split into doc spans
    for match_id, start, end in matches:
        ListOfMatchedWords.append(doc[start:end].text)
        Topic=nlpPyRank.vocab.strings[match_id]
        ListOfTopics.append(Topic)
    MatchedWords.append(", ".join(ListOfMatchedWords))
    Topics.append(", ".join(list(set(ListOfTopics))))
#Responses=[r.lower() for r in CandidateDF["Response"].to_list()]
#CandidateDF.drop('Response',inplace=True,axis=1)
#CandidateDF.insert(2,column="Response",value=Responses,allow_duplicates=True)
UniqueTopics=list(set(Topics))
TopicLabels={UniqueTopics[i]:i for i in range(0,len(UniqueTopics)) }
Participants=list(set(CandidateDF["Speaker"].to_list()))
SpeakersTargets={Participants[i]:i for i in range(0, len(Participants))}
print(TopicLabels,len(TopicLabels))
print(SpeakersTargets,len(SpeakersTargets))
CandidateDF.insert(CandidateDF.shape[1],column="Topics",value=Topics)
CandidateDF.insert(CandidateDF.shape[1],column="IntTopicTarget",value=[TopicLabels[topic] for topic in Topics])
CandidateDF.insert(CandidateDF.shape[1],column="IntSpeakerTarget",value=[SpeakersTargets[speaker] for speaker in CandidateDF["Speaker"].to_list()])
CandidateDF.insert(CandidateDF.shape[1],column="MatchedWords",value=MatchedWords)
CandidateDF.to_csv("candidate.csv")
