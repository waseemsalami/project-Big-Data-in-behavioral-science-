# project-Big-Data-in-behavioral-science-

# Big Data Analysis in Behavioral Science (Technion University)
## Summary: 
Collaborated with a psychology Master's student to analyze Reddit data, examining the motivations behind veganism and minimalism. The project involved data extraction, statistical and sentiment analysis, feature engineering, and predictive modeling using supervised machine learning techniques. Conducted statistical tests, including *ANOVA* and reliability metrics (e.g., Cronbach's Alpha, Intraclass Correlation Coefficient), to validate data and models. Generated actionable insights into behavioral motivations through rigorous data preprocessing, visualization, and predictive modeling.

## Key Skills and Tools Highlighted:

1. Data Collection & Preprocessing:
* Extracted Reddit raw data using *Reddit's API: PRAW*.

* Filtered, cleaned, and labeled data using tools like pandas and Numpy.

2. Crowdsourced Labeling:
* Designed and deployed surveys on *Amazon MTurk* to label behavioral data accurately.
* Developed classification systems for biospheric and egocentric labels.
  
3. Text Processing:
* Tokenization, stemming, and stop-word removal using *NLTK*.
* Vectorization and feature extraction with TF-IDF from scikit-learn.

4. Sentiment Analysis:
* Analyzed text sentiments with *VADER* for insights into user motivations.

5. Feature Engineering and Selection:
* Selected features through filter, wrapper, and embedded methods (e.g., Lasso, RFE).
* Engineered features like compound sentiment scores and common word frequencies.

6. Machine Learning Modeling:
* Developed models using Random Forest, Logistic Regression, and Decision Tree.
* Evaluated models with cross-validation methods like Leave-One-Out (LOO).

7. Statistical Analysis:
* Performed hypothesis testing and agreement rate analysis for data validation.
* Implemented ANOVA tests to assess group differences.
* Evaluated data reliability with Cronbach's Alpha and Intraclass Correlation Coefficient (Pingouin).
  
8. Visualization:
* Created detailed visual representations of data distributions and model results using matplotlib and seaborn.

9. Data Handling & Deployment:
* Managed large datasets for labeled and unlabeled data.
* Predicted labels for unlabeled datasets using trained models.

10. Automation and Workflow Integration:
* Automated data cleaning, processing, and validation through Python scripting.
* Combined multiple libraries for efficient workflows and analysis.

### Python Libraries Used:
Data Manipulation & Analysis: *pandas*, *NumPy*

Natural Language Processing: *NLTK*, *PRAW*, *TF-IDF*

Machine Learning: *scikit-learn (RandomForestClassifier, LogisticRegression, DecisionTreeClassifier)*

Statistical Analysis: *Pingouin, statsmodels, scipy*

Visualization: *matplotlib, seaborn*

* If you only intend to view the project you can either view the ipynb file here (on github) or download the HTML file and open it with any of your browsers, otherwise, you can download the ipynb file and surf through the code in any ipynb lab like Jupyter Notebook.

# About: intro
A python data analyzing and modeling project done during a Big Data course I took at the Technion university as a Inf Sys Eng student,
in which I teamed up as a data student with a psychology Masters Degree student, 
and together we researched a subject in behavioral science through extracting big
and rogue Reddit data, processing and modeling it, and finally evaluating and
visualizing the results while integrating statistical tools.

# About: Theory
In the last decade, two phenomena have become very popular among various populations in the Western world.
Veganism and minimalism, disseminated through movies, documentary series, and social networks,
are two behavioral acts that influence not only the individual's personal behavior but also their behavior in society.

Veganism, which means that a person does not consume any animal products,
has been extensively researched (Pendergrast, 2016, pp.106-122) 
and has a significant positive impact on the environment.

Minimalism, expressed through a preference for a life with reduced consumption and possessions (Palafox, 2021),
also constitutes pro-environmental behavior.

Research focusing on pro-environmental and pro-social behavior,
categorizes the motives of individuals for these actions into four categories (Snelgar, 2006):

1. Altruism
2. Biophilia - Animals
3. Biophilia - Plants
4. Egocentrism
    
In this Project we want to examine what motivates people to choose a vegan diet or minimize their consumption as much as possible.
Do most people choose this lifestyle out of concern for others,
concern for themselves, or concern for the environment (plants and/or animals)? 

# About: Practical

The project consists of 4 main Phases(tasks), each task has its own ipynb and html file displaying all the modeling proceess of this task.

# Task 1: Extracting the raw Reddit Posts text Data by querying with praw from relevant subreddits
I extracted the text data of the Reddit posts from various Subreddits (e.g. https://www.reddit.com/r/minimalism/ and https://www.reddit.com/r/vegan/)
and finaly I chose two relevant datasets for each group (minimalism\veganism)
and deployed a first small-batch of them to Amazon MTurk workers asking them to read the Posts of each datasets and rate to what scale 
do they think the post tends to be biospheric\egocentric(more about how its measured in the ipynb files)
* we used MTurk's ratings for deciding on the post's label: 2 for Biospheric ,1 for Egocentric and 0 for undecided
  (which was later converted to binary - 1 for Bio, 0 for Ego).
  So we actually used MTurk's ratings as our labeling and by that we formed our supervised data:
  
1. searched for relevant questions to ask the MTurk workers

2. Built a new MTurk project and designed the survey’s format

3. Found a problem in our project plan – we first chose
   to build 4 models for each label , (and fixed it) but then    
   we found a way to generalize the way we model and by
   that we got from 4 models into 2 models

4. Created a big set of questions and chose the most
   unbiased and straightforward questions
   
6. We finally deployed our chosen datasets for each of the groups: minimalism/veganism with the set of questions, in which we
   let MTurk workers label the datasets(posts) wether they think they tend to a vegan or minimalist person by reading 5 behvioral questions and rating them from (1 to 5)
 
     
# Task 2(batch_analysis):
Retrieved labled data from MTurk workers, analyzed the batch and the labeling
by different statistical and judges-agreemant rate approaches,
and finally, after some adjustments for questions that we found less relevant,
we sent the bigger batch for MTurk with updated questions list.
hence the part b of task2: Big Batch Analysis.

# Task 3(Big_batch_Analysis): here I conduct a thorough analysis of the recieved labled data:
On the Batch I performed: statistical analysis, Labeling (by deciding on a labeling method for the posts based on the scores of the answers from MTurk), Sentiment Analysis, Text Analysis and Statistical Tests on the labeling results and more.. (found on 3-Big_batch_analysis_mixed(task3).ipynb)

# Task 4-A (4-extracing_unlabled_data(task4).ipynb) : extracting unlabled data - time to dive into the world of the unknown:
 extracted a big batch of posts for each of the groups (veganism and minimalism)

# Task 4-B (5-modeling(task4).ipynb) : Modeling 
transfered labled batch to binary labeling and officialy split the labled data
into 2 datasets for further modeling: minimalism data and veganism data,
which will be split to train\test data sets for training our models.

# the main steps(which are easily shown in (5-modeling(task4).ipynb) are:
1. Feature Creation: top 90%+ common words of each dataset and compound score achieved from previous sentiment Analysis.
2. Feature Selection: performed Filter Method, Wrapper Method, Embedded Method on the feature list from previous step,
   selected features size of: 4 - 4 features seemed reasonable given the fact that our datasets are not that big.
3. Model Selection: performed with LOO(Leave One Out) method with following models: Nearest neighbors, Logistic Regression,
   Decision Tree and Random Forest.
   The model with the best auc-roc value we got on the subset of the train set is :

for veganism:RandomForestClassifier with features chosen by the embedded method :
['anim', 'motiv', 'want', 'compound']

for minimalism: RandomForestClassifier with features chosen by the wrapper method using rfe model :
['earth', 'environment', 'wast', 'compound']

4. Evaluating the chosen model on the test set : predict_proba, Leave One Out cross validation.
5. reading unlabled data
6. Creating features for the unlabled data based on features selected from labled data
7. Predicting label of unlabled data sets
8. statistical analysis

* Results are seen in the final ipynb file (modeling) in the 8th step (statistical analysis).
* A summary of the results can be also found in the pptx presentation.
	
	

