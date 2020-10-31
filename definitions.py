from spacy.matcher import PhraseMatcher,Matcher

def GetSpeakerList(dictionaryOfSpeakers={}):
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
        return dictionaryOfSpeakers;
def GetPartyDictionary():
    partydict={"OBAMA":"Democratic","MCCAIN":"Republican","ROMNEY":"Republican","MR. KENNEDY":"Democratic",
        "MR. NIXON":"Republican","MR. FORD":"Republican","MR. CARTER":"Democratic","MR. REAGAN":"Republican",
        "MR. MONDALE":"Democratic","GEORGE W. BUSH":"Republican","GEORGE H.W. BUSH":"Republican","DUKAKIS":"Democratic",
        "BILL CLINTON":"Democratic","HILLARY CLINTON":"Democratic","DOLE":"Republican","GORE":"Democratic",
        "KERRY":"Democratic","TRUMP":"Republican","BIDEN":"Democratic"}
    return partydict;
def SetFamilyMatchPatterns(fam_matcher):
    FamilyMatches=[{"LOWER":{"IN":["your","his","her"]}},{"LEMMA":{"IN":["sister","brother","family","son","daughter","wife","husband","father","dad","mom"]}}]
    fam_matcher.add("FamilyAttacks",None,FamilyMatches)
    dictionaryFoundKeywords={"FamilyAttacks":[]}
    dictionaryCounts={"FamilyAttacks":0}
    return fam_matcher,dictionaryFoundKeywords,dictionaryCounts
def SetMatchPatterns(matcher):
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
    matcher.add("Gun Control",None,[{"LOWER":"handgun"}])
    matcher.add("Gun Control",None,[{"LEMMA":"gun"},{"LOWER":"control", "OP":"?"}])
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
    matcher.add("Economy",None,[{"LOWER":"jobs"}])

    ### Social Welfare programs: e.g. Social Security, housing subsidies, federal minimum wage,
    matcher.add("Social Welfare",None,[{"LOWER":"social"},{"LOWER":"security"}])
    matcher.add("Social Welfare",None,[{"LOWER":"housing"},{"LOWER":"subsidies"}])
    matcher.add("Social Welfare",None,[{"LOWER":"minimum"},{"LOWER":"wage"}])
    matcher.add("Social Welfare",None,[{"LOWER":"prevailing"},{"LOWER":"wages"}])
    matcher.add("Social Welfare",None,[{"LOWER":"abnormal"},{"LOWER":"poverty"}])


    #### Public education
    matcher.add("Public Education",None,[{"LEMMA":{"IN":["students","teachers"]}}])
    matcher.add("Public Education",None,[{"LOWER":"public"},{"LOWER":"education"}])
    matcher.add("Public Education",None,[{"LEMMA":"school"},{"LOWER":{"IN":["buildings","teachers","districts","violence"],"OP":"?"}}])
    matcher.add("Public Education",None,[{"LOWER":"teacher"},{"LOWER":"salaries"}])
    matcher.add("Public Education",None,[{"LOWER":"high"},{"LOWER":"school"},{"LOWER":"graduates"}])

    #### Abortion
    matcher.add("abortion",None,[{"LOWER":{"IN":["abortions","abortion"]}}])
    matcher.add("abortion",None,[{"LOWER":{"IN":["anti","pro"]}},{"IS_PUNCT":True,"OP":"?"},{"LEMMA":"abortion"}])

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
    matcher.add("Immigration",None,[{"LOWER":"immigration"}])
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


    matcher.add("Taxes",None,[{"Lemma":"tax"}, {"LOWER":{"IN":["credit","credits","penalties","provisions","cuts"], "OP":"?"}}])
    matcher.add("Taxes",None,[{"LOWER":{"IN":["raise","lower","increase","decrease"]}},{"Lemma":"tax"}])
    matcher.add("Taxes",None,[{"LOWER":"income"},{"LEMMA":"tax"}])

    #### Return these
    dictionaryFoundKeywords={"Race":[],"Immigration":[], "Gun Control":[],"Climate Change":[], "Federal Spending":[], "abortion":[], "National Defense":[],
    "Oil Industry":[], "Economy":[], "Public Education":[],"Health Care":[], "Social Welfare":[],"Taxes":[]}
    dictionaryCounts={"Race":0,"Immigration":0, "Gun Control":0,"Climate Change":0, "Federal Spending":0, "abortion":0, "National Defense":0,
    "Oil Industry":0, "Economy":0, "Public Education":0,"Health Care":0, "Social Welfare":0, "Taxes":0}
    return matcher,dictionaryFoundKeywords,dictionaryCounts

def TimestampsForDebates():
    TimeStamps={"Kennedy-Nixon":1960,"Carter-Ford":1976, "Carter-Reagan":1980,"Reagan-Mondale":1984,
    "Bush-Dukakis":1988, "Clinton-Bush":1992, "Clinton-Dole":1996, "Gore-Bush":2000, "Bush-Kerry":2004,
    "McCain-Obama":2008, "Obama-Romney":2012, "Clinton-Trump":2016, "JoeBidenVDonaldTrump":2020}
    return TimeStamps
