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

from definitions import GetPartyDictionary,SetMatchPatterns,SetFamilyMatchPatterns,TimestampsForDebates

matplotlib.rc('font',family='monospace')
plt.style.use('ggplot')
fig, ax = plt.subplots()
plt.xlabel("Debate Topics")
plt.ylabel("Count Freq.")


def BuildDF(partydict,DoAll=False,ADebate="",Relabel=True):
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
    if Relabel:party=[PartyInt[p] for p in party]
    FascilatorDF=Debate_df[Debate_df["Candidates"]==False]
    CandidateDF.drop('Candidates',inplace=True,axis=1)
    FascilatorDF.drop('Candidates',inplace=True,axis=1)
    del Debate_df
    CandidateDF.insert(CandidateDF.shape[1],"party",party,True)
    return CandidateDF,FascilatorDF
def CreateScatterText(CandidateDF):
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
    ).build()#.compact(AssociationCompactor(compact, use_non_text_features=True))
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
def InputTextFeatures(nlpPyRank,PromptsModeratorComments,threshold,stop_words):
    docs=nlpPyRank.pipe(PromptsModeratorComments)
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

##### These lines are only for feature exploration
def BuildTopics(nlpPyRank,stop_words,PromptsModeratorComments,RankScore=0.1,NTopics=6,TopicWords=20,doLDA=False):
    scikittext=InputTextFeatures(nlpPyRank,PromptsModeratorComments,RankScore,stop_words)
    model,count_feature_names=CreateTopics(scikittext,stop_words,NTopics,doLDA)
    topicWords=display_topics(model,count_feature_names,TopicWords)
    return model,topicWords

def PlotScatterText(partydict):
    CandidateDF,FascilatorDF=BuildDF(partydict,DoAll=True,Relabel=False)#### All DataFrame
    del FascilatorDF;
    CandidateDF=CandidateDF.sample(frac=1.)
    CreateScatterText(CandidateDF)


#### Build for pie charts:


#PromptsModeratorComments=[p.lower() for p in PromptsModeratorComments]
#### Mode for running Scattertext
'''
partydict=GetPartyDictionary()
nlpPyRank = spacy.load("en_core_web_lg")
tr = pytextrank.TextRank()
nlpPyRank.add_pipe(tr.PipelineComponent, name="textrank", last=True)
matcher = Matcher(nlpPyRank.vocab)

stop_words=['evening','night','senator','vice','president','united','states','question','questions','seconds','second','minutes','minute','last','mr','time','governor','closing','opening','rebuttal','segment',"follow up","transcription","town hall"]

#### For all topics
CandidateDF,FascilatorDF=BuildDF(partydict,DoAll=True)
del CandidateDF
Moderatorlist=list(set(FascilatorDF["Speaker"].to_list()))
del FascilatorDF
speakers=[key.lower() for key in partydict.keys() ]
stop_words.extend(", ".join(speakers).split(" "))
stop_words.extend(", ".join(Moderatorlist).split())
for w in stop_words:nlpPyRank.vocab[w].is_stop = True;
DebatesandTimestamps=TimestampsForDebates()
for key in DebatesandTimestamps.keys():
    CandidateDF,FascilatorDF=BuildDF(partydict,False,key)#### All DataFrame
    del CandidateDF;
    PromptsModeratorComments=FascilatorDF["Response"].to_list()
    BuildTopics(PromptsModeratorComments)

'''


def FillTopicCounts(nlpPyRank,matcher,Canddocs,DebateTitle,dictionaryCounts,dictionaryFoundKeywords):
    TopicMatches=[]
    for doc in Canddocs:### Loop in order of the response column
        matches=matcher(doc)
        if(len(matches)>0):TopicMatches.append(True)
        else:TopicMatches.append(False)
        for match_id, start, end in matches:
            Topic=nlpPyRank.vocab.strings[match_id]
            dictionaryCounts[Topic]=dictionaryCounts[Topic]+1#### Count for each topic
            dictionaryFoundKeywords[Topic].append(doc[start:end].text)
    for key in dictionaryFoundKeywords.keys():dictionaryFoundKeywords[key]=list(set(dictionaryFoundKeywords[key]))#### Keywords
    Topics=[key for key in dictionaryCounts.keys()]
    Counts=[dictionaryCounts[key] for key in dictionaryCounts.keys()]
    Keywords=[", ".join(dictionaryFoundKeywords[key]) for key in dictionaryCounts.keys()]
    Dfout=pd.DataFrame.from_dict({"Topic":Topics,"Count":Counts,"Keywords":Keywords})
    print(Counts)
    return Dfout,TopicMatches



def BuildPieCharts(nlpPyRank,DebatesandTimestamps,matcher):
    matcher,dictionaryFoundKeywords,dictionaryCounts=SetMatchPatterns(matcher)
    partydict=GetPartyDictionary()
    ListForPieCharts=[]
    for key in DebatesandTimestamps.keys():
        CandidateDF,FascilatorDF=BuildDF(partydict,False,key)#### All DataFrame
        del FascilatorDF
        CandResponses=CandidateDF["Response"].to_list()
        Canddocs=nlpPyRank.pipe(CandResponses)
        for TopicKey in dictionaryCounts:
            dictionaryCounts[TopicKey]=0
            dictionaryFoundKeywords[TopicKey]=[]
        OutputDFPie,TopicMatches=FillTopicCounts(nlpPyRank,matcher,Canddocs,key,dictionaryCounts,dictionaryFoundKeywords)
        Total=len(CandResponses)
        Matches=sum(TopicMatches)
        OutputDFPie.insert(OutputDFPie.shape[1],column="Year",value=[DebatesandTimestamps[key] for i in range(OutputDFPie.shape[0])])
        OutputDFPie.insert(OutputDFPie.shape[1],column="Debate",value=[key for i in range(OutputDFPie.shape[0])])
        OutputDFPie.insert(OutputDFPie.shape[1],column="MatchEff",value=[Matches/Total for i in range(OutputDFPie.shape[0])])
        ListForPieCharts.append(OutputDFPie)
    PieCharts=pd.concat(ListForPieCharts)
    PieCharts.to_csv("PieChartsPerYear.csv")

def ParseBasedOnTokenLength(nlpPyRank,CandidateDF):
    CandResponses=CandidateDF["Response"].to_list()
    CandParty=CandidateDF["party"].to_list()
    Speakers=CandidateDF["Speaker"].to_list()
    Canddocs=nlpPyRank.pipe(CandResponses)
    dictionaryParse={"Speaker":[],"Response":[],"party":[]}
    count=0
    #### this would be the parser
    for doc in Canddocs:
        if len(doc)>192:
            #print(doc.text,len(doc.text))
            divisions=int(len(doc)/192)
            for i in range(divisions):
                start=i*192
                end=(i+1)*192
            #print(start,end,doc[start:end].text)
                dictionaryParse["Speaker"].append(Speakers[count])
                dictionaryParse["party"].append(CandParty[count])
                dictionaryParse["Response"].append(doc[start:end].text)
            if not len(doc)%192==0:
                end=divisions*192
                dictionaryParse["Response"].append(doc[end:len(doc)].text)
                dictionaryParse["Speaker"].append(Speakers[count])
                dictionaryParse["party"].append(CandParty[count])
        else:
            dictionaryParse["Response"].append(doc.text)
            dictionaryParse["Speaker"].append(Speakers[count])
            dictionaryParse["party"].append(CandParty[count])
        count=count+1
    CandidateDF=pd.DataFrame(dictionaryParse)
    return CandidateDF
'''
BuildPieCharts(DebatesandTimestamps,matcher)
'''


def TotalCandPieChart(nlpPyRank,matcher,Canddocs):
    matcher,dictionaryFoundKeywords,dictionaryCounts=SetMatchPatterns(matcher)
    TopicMatches=[]
    for doc in Canddocs:### Loop in order of the response column
        matches=matcher(doc)
        TopicMatches.append(len(matches)>0)
        for match_id, start, end in matches:
            Topic=nlpPyRank.vocab.strings[match_id]
            dictionaryCounts[Topic]=dictionaryCounts[Topic]+1#### Count for each topic
            dictionaryFoundKeywords[Topic].append(doc[start:end].text)
    for key in dictionaryFoundKeywords.keys():dictionaryFoundKeywords[key]=list(set(dictionaryFoundKeywords[key]))#### Keywords
    Topics=[key for key in dictionaryCounts.keys()]
    Counts=[dictionaryCounts[key] for key in dictionaryCounts.keys()]
    Keywords=[", ".join(dictionaryFoundKeywords[key]) for key in dictionaryCounts.keys()]
    Dfout=pd.DataFrame.from_dict({"Topic":Topics,"Count":Counts,"Keywords":Keywords})
    Dfout.to_csv("TotalYieldsPieChart.csv")
    return TopicMatches,matcher
#### return TopicMatches and CandidateDF
####Create train and test sets Input Topics Input CandidateDF void
#TopicMatches=TotalCandPieChart(matcher,Canddocs)

#### Make this a function that returns Test, Labeled samples
'''
CandidateDF,FascilatorDF=BuildDF(partydict,True)
del FascilatorDF;
print(CandidateDF.head())
CandidateDF=ParseBasedOnTokenLength(CandidateDF)
CandResponses=CandidateDF["Response"].to_list()
CandParty=CandidateDF["party"].to_list()
Speakers=CandidateDF["Speaker"].to_list()
Canddocs=nlpPyRank.pipe(CandResponses)
Participants=list(set(Speakers))
SpeakersTargets={Participants[i]:i for i in range(0, len(Participants))}
print(SpeakersTargets,len(SpeakersTargets))
CandidateDF.insert(CandidateDF.shape[1],column="IntSpeakerTarget",value=[SpeakersTargets[speaker] for speaker in CandidateDF["Speaker"].to_list()])
#### This would be the start of a function that takes in matcher, canddocs
Canddocs=nlpPyRank.pipe(CandidateDF.Response.to_list())
'''
def CreateTrainTestSamples(nlpPyRank,matcher,TopicMatches,CandidateDF):

    CandidateDF.insert(CandidateDF.shape[1],column="MatchToTopic",value=TopicMatches)
    UnlabeledCandidateDF=CandidateDF[CandidateDF.MatchToTopic==False]
    UnlabeledCandidateDF.to_csv("Test_candidates.csv")
    CandidateDF=CandidateDF[CandidateDF.MatchToTopic==True]
    CandDocsKeywords=nlpPyRank.pipe(CandidateDF.Response.to_list())
    MatchedWords=[]
    Topics=[]
    for doc in CandDocsKeywords:### Loop in order of the response column
        matches=matcher(doc)
        ListOfMatchedWords=[]
        ListOfTopics=[]
        #### need to limit doc size for BERT to up to 512 tokens
        for match_id, start, end in matches:
            ListOfMatchedWords.append(doc[start:end].text)
            Topic=nlpPyRank.vocab.strings[match_id]
            ListOfTopics.append(Topic)
        MatchedWords.append(", ".join(ListOfMatchedWords))
        Topics.append(", ".join(list(set(ListOfTopics))))
    UniqueTopics=list(set(Topics))
    TopicLabels={UniqueTopics[i]:i for i in range(0,len(UniqueTopics)) }
    CandidateDF.insert(CandidateDF.shape[1],column="Topics",value=Topics)
    CandidateDF.insert(CandidateDF.shape[1],column="IntTopicTarget",value=[TopicLabels[topic] for topic in Topics])
    CandidateDF.insert(CandidateDF.shape[1],column="MatchedWords",value=MatchedWords)
    #### Need to parse into 128 token segments so that I don't blow up RAM and the code fits on GPU
    CandidateDF.to_csv("candidate.csv")
