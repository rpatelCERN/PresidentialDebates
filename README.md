# PresidentialDebates
A NLP project looking at 60 years of presidential debates

## Main steps:
The main analyzer function is DebateAnalyzer.py which has four options which can run independantly

 ``` --scattertext ### create an html for an interactive scattertext plot  ``` 

 ``` --TopicKeyWords ### Use NMF, CountVectorizer, to extract keywords from each set of debates ```

 ``` --PieChartInput ### Use the SpaCy matcher to find matches to keywords and fill a dictionary of counts ```

 ``` --TrainTestSamples #### Create csv files for testing, validating and training a BERT text classification model  ```

The results can be seen [In this Medium article](https://rgp230.medium.com/in-their-own-words-60-years-of-presidential-debates-7cb4cc40e92c)


