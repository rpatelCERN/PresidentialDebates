# PresidentialDebates
A NLP project looking at 60 years of presidential debates

## Main steps:
The webscrape is performed with WebScrapeDebates.py using beautiful soup 4

The main analyzer function is DebateAnalyzer.py which has four options which can run independantly

 ``` --scattertext ### create an html for an interactive scattertext plot  ``` 

 ``` --TopicKeyWords ### Use NMF, CountVectorizer, to extract keywords from each set of debates ```

 ``` --PieChartInput ### Use the SpaCy matcher to find matches to keywords and fill a dictionary of counts ```

 ``` --TrainTestSamples #### Create csv files for testing, validating and training a BERT text classification model  ```

The library of worker functions are contained in TrainingInputAndFeatures.py and definitions.py contains dictionaries for speakers and parties and SpaCy match patterns. 

The results can be seen [In this Medium article](https://rgp230.medium.com/in-their-own-words-60-years-of-presidential-debates-7cb4cc40e92c)

## Main contents of the Repo

* HTML/ contains the raw HTML data of the debates

* RawCSV/ contains the CSV files formatted from the raw HRML using WebScrapeDebates.py

* ValidationSet/ and LabeledCSV/ contain the training and testing datasets that are input to the [GoogleCoLab ML notebook](https://colab.research.google.com/drive/1p63jfnXagqBKx_X9hD0_PFsYlWeYWAsG?usp=sharing). Also I copied over the test sample with the classifier weights to LabeledCSV/

* Scattertext plot is hosted on [github.io](https://rpatelcern.github.io/PresidentialDebates/pytextrank_rankdiff_feature_exploration.html)

