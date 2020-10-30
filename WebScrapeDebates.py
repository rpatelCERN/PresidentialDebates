from bs4 import BeautifulSoup
import urllib3
import requests
import re
import pandas as pd
import glob,os
def PrettifyHTMLScrape(weburl):
    req = requests.get(weburl)
    htmlSource=req.content
    soup = BeautifulSoup(htmlSource,"html5lib")
    soup.prettify()
    return soup.get_text()

def GrabDebateURLs(weburl):
    req = requests.get(weburl)
    soup = BeautifulSoup(req.content, 'html.parser')
    #print(soup.find('p').get_text())
    linksToTranscripts=[]
    header=soup.find("h1")
    for sib in header.find_all_next():
        if sib.get("href") is None:continue
        if ("Presidential Debate" in sib.text) and not("Vice Presidential Debate" in sib.text):
            linksToTranscripts.append((sib.text,'https://www.debates.org'+sib.get('href')))
    linksToTranscripts.append(("October 11, 1992: The First Clinton-Bush-Perot Presidential Debate First Half",'https://www.debates.org/voter-education/debate-transcripts/october-11-1992-first-half-debate-transcript/'))
    linksToTranscripts.append(("October 11, 1992: The First Clinton-Bush-Perot Presidential Debate Second Half",'https://www.debates.org/voter-education/debate-transcripts/october-11-1992-second-half-debate-transcript/'))
    linksToTranscripts.append(("October 15, 1992: The Second Clinton-Bush-Perot Presidential Debate First Half", 'https://www.debates.org/voter-education/debate-transcripts/october-15-1992-first-half-debate-transcript/'))
    linksToTranscripts.append(("October 15, 1992: The Second Clinton-Bush-Perot Presidential Debate Second Half", 'https://www.debates.org/voter-education/debate-transcripts/october-15-1992-second-half-debate-transcript/'))
    return linksToTranscripts
def BoilSoup(weburl="https://www.debates.org/voter-education/debate-transcripts"):
    Links=GrabDebateURLs(weburl)
    FileNames=[]
    for title,link in Links:
        filename=title.split(": ")[1].replace(" ","_")
        souptext=PrettifyHTMLScrape(link)
        #Date=title.split(": ")[0]+" Debate Transcript"
        #print(Date)
        #print(souptext)### Gets
        souptext=" ".join(souptext.split('\n'))
        #print()
        fout=open("HTML/"+filename+".html",'w')
        fout.seek(0)
        fout.write(souptext)
        fout.close()
        FileNames.append(filename+".html")
        print(filename)
    return FileNames


def CreateRawCSV():
    FileNames=BoilSoup();
    ###From USA Today transcripts
    FileNames.append("JoeBidenVDonaldTrump1.html")
    FileNames.append("JoeBidenVDonaldTrump2.html")
    ### Move these to a definitions file
    #### 4 debates JFK v Nixon 1960
    #### Put a function in defintions to create this dictionary
    dictionaryOfSpeakers={"The_First_Kennedy-Nixon_Presidential_Debate.html":["MR. NIXON", "MR. KENNEDY","SENATOR KENNEDY", "MR. SMITH"]}
    dictionaryOfSpeakers.update({"The_Second_Kennedy-Nixon_Presidential_Debate.html":["MR. NIXON", "MR. KENNEDY","MR. McGEE","MR. NIVEN","MR. MORGAN","MR. SPIVAK","MR. LEVY"]})
    dictionaryOfSpeakers.update({"The_Third_Kennedy-Nixon_Presidential_Debate.html":["MR. NIXON", "MR. KENNEDY","MR. SHADEL","MR. McGee","MR. DRUMMOND","MR. VON FREMD"]})
    dictionaryOfSpeakers.update({"The_Fourth_Kennedy-Nixon_Presidential_Debate.html":["MR. NIXON", "MR. KENNEDY","MR. HOWE"]})
    ####  3 debates Carter v. Ford 1976
    dictionaryOfSpeakers.update({"The_First_Carter-Ford_Presidential_Debate.html":["MR. FORD","MR. CARTER","MR. NEWMAN","MR. GANNON","MR. REYNOLDS","MS. DREW"]})
    dictionaryOfSpeakers.update({"The_Second_Carter-Ford_Presidential_Debate.html":["MR. FORD","MR. CARTER","MS. FREDERICK","MR. FRANKEL","MR. TREWHITT","MR. VALERIANI"]})
    dictionaryOfSpeakers.update({"The_Third_Carter-Ford_Presidential_Debate.html":["MR. FORD","MR. CARTER","MS. WALTERS","MR. KRAFT","MR. MAYNARD","MR. NELSON"]})

    #### 1 debates Carter v. Reagen 1980
    dictionaryOfSpeakers.update({"The_Anderson-Reagan_Presidential_Debate.html":["REAGAN","MOYERS","ANDERSON"]})
    dictionaryOfSpeakers.update({"The_Carter-Reagan_Presidential_Debate.html":["MR. REAGAN","MR. CARTER","MR. SMITH","MR. STONE","MR. ELLIS","MR. HILLIARD","MS. WALTERS"]})
    #### 2 debates Mondale v. Reagan 1984
    dictionaryOfSpeakers.update({"The_First_Reagan-Mondale_Presidential_Debate.html":["THE PRESIDENT","MR. MONDALE","MS. WALTERS","MR. WIEGHART","MS. SAWYER","MR. BARNES"]})### This file has some mis-formatting in the beginning
    dictionaryOfSpeakers.update({"The_Second_Reagan-Mondale_Presidential_Debate.html":["THE PRESIDENT","MR. MONDALE","MR. NEWMAN","MS. GEYER","MR. KALB","MR. KONDRACKE","Mr. Trewhitt"]})### Reformat the last panelist

    #### 2 debates Dukakis v. Bush 1988
    dictionaryOfSpeakers.update({"The_First_Bush-Dukakis_Presidential_Debate.html":["BUSH","DUKAKIS","LEHRER","JENNINGS","MASHEK"]})
    dictionaryOfSpeakers.update({"The_Second_Bush-Dukakis_Presidential_Debate.html":["BUSH","DUKAKIS","SHAW","COMPTON","WARNER"]})
    #### 3 debates split by halfs Clinton v.Bush 1992
    dictionaryOfSpeakers.update({"The_First_Clinton-Bush-Perot_Presidential_Debate_First_Half.html":["PEROT","PRESIDENT BUSH","CLINTON","MASHEK","LEHRER","VANOCUR","COMPTON"]})
    dictionaryOfSpeakers.update({"The_First_Clinton-Bush-Perot_Presidential_Debate_Second_Half.html":["PEROT","BUSH","CLINTON","MASHEK","LEHRER","VANOCUR","COMPTON"]})
    dictionaryOfSpeakers.update({"The_Second_Clinton-Bush-Perot_Presidential_Debate_First_Half.html":["PEROT","BUSH","CLINTON","SIMPSON","AUDIENCE QUESTION"]})
    dictionaryOfSpeakers.update({"The_Second_Clinton-Bush-Perot_Presidential_Debate_Second_Half.html":["PEROT","BUSH","CLINTON","SIMPSON","AUDIENCE QUESTION"]})
    dictionaryOfSpeakers.update({"The_Third_Clinton-Bush-Perot_Presidential_Debate.html":["PEROT","BUSH","CLINTON","LEHRER","THOMAS","GIBBONS","ROOK"]})

    ####  2 debates Clinton v. Dole 1996
    dictionaryOfSpeakers.update({"The_First_Clinton-Dole_Presidential_Debate.html":["CLINTON","DOLE","LEHRER"]})
    dictionaryOfSpeakers.update({"The_Second_Clinton-Dole_Presidential_Debate.html":["CLINTON","DOLE","LEHRER","MR.","MS.","DR."]})#### Needs more formatting for audience questions

    #### 3 debates Gore v. Bush 2000
    dictionaryOfSpeakers.update({"The_First_Gore-Bush_Presidential_Debate.html":["GORE","BUSH","MODERATOR"]})
    dictionaryOfSpeakers.update({"The_Second_Gore-Bush_Presidential_Debate.html":["GORE","BUSH","MODERATOR"]})
    dictionaryOfSpeakers.update({"The_Third_Gore-Bush_Presidential_Debate.html":["GORE","BUSH","MODERATOR","MEMBER OF AUDIENCE"]})

    ####3 debates Kerry v. Bush 2004
    dictionaryOfSpeakers.update({"The_First_Bush-Kerry_Presidential_Debate.html":["KERRY","BUSH","LEHRER"]})
    dictionaryOfSpeakers.update({"The_Second_Bush-Kerry_Presidential_Debate.html":["KERRY","BUSH","GIBSON"]})### AUDIENCE questions by NAME merged with moderator
    dictionaryOfSpeakers.update({"The_Third_Bush-Kerry_Presidential_Debate.html":["KERRY","BUSH","SCHIEFFER"]})

    #### 3 debates Obama v. McCain 2008
    dictionaryOfSpeakers.update({"The_First_McCain-Obama_Presidential_Debate.html":["MCCAIN","OBAMA","LEHRER"]})
    dictionaryOfSpeakers.update({"The_Second_McCain-Obama_Presidential_Debate.html":["MCCAIN","OBAMA","BROKAW"]})
    dictionaryOfSpeakers.update({"The_Third_McCain-Obama_Presidential_Debate.html":["MCCAIN","OBAMA","SCHIEFFER"]})


    #### 3 debates Obama v. Romney 2012
    dictionaryOfSpeakers.update({"The_First_Obama-Romney_Presidential_Debate.html":["OBAMA","ROMNEY","LEHRER"]})
    dictionaryOfSpeakers.update({"The_Second_Obama-Romney_Presidential_Debate.html":["OBAMA","ROMNEY","CROWLEY","QUESTION"]})
    dictionaryOfSpeakers.update({"The_Third_Obama-Romney_Presidential_Debate.html":["OBAMA","ROMNEY","SCHIEFFER"]})

    #### 3 debates Clinton v. Trump 2016
    dictionaryOfSpeakers.update({"The_First_Clinton-Trump_Presidential_Debate.html":["CLINTON","TRUMP","HOLT"]})
    dictionaryOfSpeakers.update({"The_Second_Clinton-Trump_Presidential_Debate.html":["CLINTON","TRUMP","COOPER","RADDATZ","QUESTION"]})
    dictionaryOfSpeakers.update({"The_Third_Clinton-Trump_Presidential_Debate.html":["CLINTON","TRUMP","WALLACE"]})

    dictionaryOfSpeakers.update({"JoeBidenVDonaldTrump1.html":["BIDEN","TRUMP","WALLACE"]})
    dictionaryOfSpeakers.update({"JoeBidenVDonaldTrump2.html":["Biden","Trump","Welker"]})
    path="HTML/"
    #FileNames=["JoeBidenVDonaldTrump1.html"]
    #print(dictionaryOfSpeakers)
    offset=0
    for f in FileNames:
        if f=="JoeBidenVDonaldTrump2.html":offset=10
        if f=="JoeBidenVDonaldTrump1.html":offset=8

        #fin=open(f,'r')
        Participant=[]
        Transcript=[]
        fullpath=path+f
        with open(fullpath) as fin:
            Speakers=dictionaryOfSpeakers[f]
            #print(f,Speakers)
            lines=(l for l in fin if any(name in l for name in Speakers))
            #lines=(l for l in fin if not "CPD" in l)### Chomp header/footer html
            for l in lines:
                if f=="JoeBidenVDonaldTrump1.html":
                    for s in Speakers:l=l.replace(s, "%s: " %s)### For consistency
                delimiters=[Speaker+": " for Speaker in Speakers]
                regexPattern = '|'.join(map(re.escape,delimiters))
                DebateFrame=re.split("(%s)" %regexPattern, l)
                DebateFrame.pop(0)
                tail = DebateFrame[-1:]
                if len(tail)>0:
                    if '©' in tail[0]:DebateFrame[-1]=tail[0].split('©')[0]
                for i in range(0,len(DebateFrame)-1):
                    if i%2==0:
                        DebateQuote=DebateFrame[i]+DebateFrame[i+1]
                        #print(DebateQuote)
                        responses=DebateQuote.split(": ")
                        Participant.append(responses[0])
                        Quote=": ".join(responses[1:len(responses)])### Do it this way in case the debate response transcript contains ":"
                        #print(Quote[0:len(Quote)-offset])
                        Transcript.append(Quote.lower())
        df = pd.DataFrame(list(zip(Participant,Transcript)),columns=['Speaker','Response'])
        df=df[~(df.Response.isnull())]
        if("Reagan-Mondale" in f):df["Speaker"]=df["Speaker"].replace("THE PRESIDENT","MR. REAGAN")### For consistency
        if("Kennedy-Nixon" in f):df["Speaker"]=df["Speaker"].replace("SENATOR KENNEDY","MR. KENNEDY")### For consistency

        df.to_csv("RawCSV/"+f.replace("html","csv"))
    DebatesToDisambiguate=["Clinton-Trump","Clinton-Bush","Gore-Bush","Bush-Kerry"]
    listOfCSV=glob.glob(os.path.join('', "*.csv" ))

    FilesToModify=[ file for file in listOfCSV if any(debatename in file for debatename in DebatesToDisambiguate)]

    for f in FilesToModify:
        Debate_df=pd.read_csv(f,index_col=0)
        if "Clinton-Trump" in f: Debate_df.Speaker=Debate_df.Speaker.replace("CLINTON","HILLARY CLINTON")
        if "Gore-Bush" in f or "Bush-Kerry" in f: Debate_df.Speaker=Debate_df.Speaker.replace("BUSH","GEORGE W. BUSH")
        if "Clinton-Bush" in f:
            Debate_df.Speaker=Debate_df.Speaker.replace("CLINTON","BILL CLINTON")
            if "First_Half" in f and "The_First" in f:Debate_df.Speaker=Debate_df.Speaker.replace("PRESIDENT BUSH","GEORGE H.W. BUSH")
            else: Debate_df.Speaker=Debate_df.Speaker.replace("BUSH","GEORGE H.W. BUSH")
        #Debate_df = Debate_df.loc[:, ~Debate_df.columns.str.contains('^Unnamed')]
        print(Debate_df.head())

        Debate_df.to_csv("RAWCSV/"+f)
CreateRawCSV();
