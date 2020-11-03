from TrainingInputAndFeatures import *
import argparse

parser = argparse.ArgumentParser(description="CORD Crusher processes for the CORD 19 dataset")
parser.add_argument("--scattertext", action='store_true', dest="scattertext")
parser.add_argument("--TopicKeyWords", action='store_true', dest="TopicKeyWords")
parser.add_argument("--PieChartInput", action='store_true', dest="PieChartInput")
parser.add_argument("--TrainTestSamples", action='store_true', dest="TrainTestSamples")

partydict=GetPartyDictionary()
nlpPyRank = spacy.load("en_core_web_lg")
tr = pytextrank.TextRank()
nlpPyRank.add_pipe(tr.PipelineComponent, name="textrank", last=True)
matcher = Matcher(nlpPyRank.vocab)
stop_words=['evening','night','senator','vice','president','united','states','question','questions',
'seconds','second','minutes','minute','last','mr','time','governor','closing','opening','rebuttal',
'segment',"follow up","transcription","town hall"]
DebatesandTimestamps=TimestampsForDebates()

args = parser.parse_args()
if args.scattertext:
    PlotScatterText(partydict)
if args.TopicKeyWords:
    #### For all topics
    CandidateDF,FascilatorDF=BuildDF(partydict,DoAll=True)
    del CandidateDF
    Moderatorlist=list(set(FascilatorDF["Speaker"].to_list()))
    del FascilatorDF
    speakers=[key.lower() for key in partydict.keys() ]
    stop_words.extend(", ".join(speakers).split(" "))
    stop_words.extend(", ".join(Moderatorlist).split())
    for w in stop_words:nlpPyRank.vocab[w].is_stop = True;
    for key in DebatesandTimestamps.keys():
        CandidateDF,FascilatorDF=BuildDF(partydict,False,key)#### All DataFrame
        del CandidateDF;
        PromptsModeratorComments=FascilatorDF["Response"].to_list()
        BuildTopics(nlpPyRank,stop_words,PromptsModeratorComments)
    del CandidateDF,FascilatorDF
if args.PieChartInput:BuildPieCharts(nlpPyRank,DebatesandTimestamps,matcher)
if args.TrainTestSamples:
    CandidateDF,FascilatorDF=BuildDF(partydict,True)
    print(CandidateDF.head())
    del FascilatorDF;
    CandidateDF=ParseBasedOnTokenLength(nlpPyRank,CandidateDF)
    print(CandidateDF.head())
    CandResponses=CandidateDF["Response"].to_list()
    #CandParty=CandidateDF["party"].to_list()
    Speakers=CandidateDF["Speaker"].to_list()
    Canddocs=nlpPyRank.pipe(CandResponses)
    Participants=list(set(Speakers))
    SpeakersTargets={Participants[i]:i for i in range(0, len(Participants))}
    print(SpeakersTargets,len(SpeakersTargets))
    CandidateDF.insert(CandidateDF.shape[1],column="IntSpeakerTarget",value=[SpeakersTargets[speaker] for speaker in CandidateDF["Speaker"].to_list()])
    #### This would be the start of a function that takes in matcher, canddocs
    Canddocs=nlpPyRank.pipe(CandidateDF.Response.to_list())
    TopicMatches,matcher=TotalCandPieChart(nlpPyRank,matcher,Canddocs)
    CreateTrainTestSamples(nlpPyRank,matcher,TopicMatches,CandidateDF)
