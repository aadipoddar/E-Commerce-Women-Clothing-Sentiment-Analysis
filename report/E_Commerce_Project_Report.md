# A PROJECT REPORT

## On

# A COMPARATIVE EVALUATION OF TEN MACHINE LEARNING CLASSIFIERS FOR SENTIMENT CLASSIFICATION OF ONLINE WOMEN'S CLOTHING REVIEWS USING SPARSE AND DENSE TEXT FEATURES

## Submitted to
### KIIT Deemed to be University

### In Partial Fulfilment of the Requirement for the Award of
## BACHELOR'S DEGREE IN COMPUTER SCIENCE AND ENGINEERING

**BY**

Prerona Karmakar — 23051689

Aadi Poddar — 23051641

**Under the guidance of**

Dr. Mahendra Kumar Gourisaria

Associate Professor

---

### SCHOOL OF COMPUTER ENGINEERING
### KALINGA INSTITUTE OF INDUSTRIAL TECHNOLOGY
### BHUBANESWAR, ODISHA - 751024

**Session: 2026-2027**

---

\newpage

### KIIT Deemed to be University
### School of Computer Engineering
### Bhubaneswar, Odisha - 751024

## CERTIFICATE

This is to certify that the report entitled **"A COMPARATIVE EVALUATION OF TEN MACHINE LEARNING CLASSIFIERS FOR SENTIMENT CLASSIFICATION OF ONLINE WOMEN'S CLOTHING REVIEWS USING SPARSE AND DENSE TEXT FEATURES"**

Submitted by:

- Prerona Karmakar — 23051689
- Aadi Poddar — 23051641

is a record of bonafide work carried out by them, in the partial fulfilment of the requirement for the award of Degree of Bachelor of Computer Science & Engineering at KIIT Deemed to be University, Bhubaneswar. This work is done during the year 2026-2027, under our guidance.

Date: ____ / ____ / ________

---

Dr. Mahendra Kumar Gourisaria

Associate Professor

Project Guide

---

\newpage

## DECLARATION

We do hereby declare that this report entitled **"A COMPARATIVE EVALUATION OF TEN MACHINE LEARNING CLASSIFIERS FOR SENTIMENT CLASSIFICATION OF ONLINE WOMEN'S CLOTHING REVIEWS USING SPARSE AND DENSE TEXT FEATURES"**, submitted by Prerona Karmakar (23051689) and Aadi Poddar (23051641), in partial fulfilment of the requirement for the award of the degree of Bachelor of Technology in Computer Science and Engineering at the School of Computer Engineering, Kalinga Institute of Industrial Technology (KIIT), Deemed to be University, Bhubaneswar, is our own work and has not been submitted to any other institute.

All the data, results and figures reported here were produced by us during the project period. Any material taken from published work has been cited in the bibliography.

Prerona Karmakar — Roll No.: 23051689

Aadi Poddar — Roll No.: 23051641

Place: Bhubaneswar

Date: ____ / ____ / ________

---

\newpage

## ACKNOWLEDGEMENTS

We wish to extend our heartfelt thanks to our supervisor, Dr. Mahendra Kumar Gourisaria, whose guidance, expertise and steady support were instrumental in the successful completion of this project. His feedback at every review helped us keep the work focused on measurable results rather than on numbers that only look good on paper.

We are also grateful to the School of Computer Engineering for providing the computing facilities and the academic environment in which this study was carried out. The entire project was completed on ordinary laptop hardware, and that was possible only because the department encouraged us to design a study that a normal machine could actually finish.

We extend our appreciation to our founder, Dr. Achyuta Samanta, for furnishing the essential resources and facilities that allowed us to carry out this research work.

We also thank each other. This project was built by two people who split the work, argued about the metrics, and repeated experiments when the first answer looked too convenient. That process made the final report stronger than either of us would have produced alone.

Finally, we appreciate the understanding, patience and encouragement of our families and friends throughout this work.

Sincerely,

Prerona Karmakar — 23051689

Aadi Poddar — 23051641

---

\newpage

## ABSTRACT

Online clothing stores collect thousands of written reviews from their customers, and reading all of them by hand is not practical for any team. This project studies how well machine learning can decide whether a clothing review is positive or negative using only the text the customer wrote. The data is the public Women's E-Commerce Clothing Reviews collection, which holds 23,486 reviews written by real customers of a United States retailer.

The first part of the work is a careful cleaning stage. Rows with no review body were dropped, short titles were joined to the review body so that headlines like "Runs small" were not thrown away, and exact duplicate texts were removed. The most important cleaning decision was the removal of all three-star reviews. We did not remove them because it was convenient. We measured them first. Among three-star reviewers, only 41.41 percent said they would recommend the product, which is close to a coin toss, while one-star reviewers rejected the product 98.10 percent of the time and five-star reviewers recommended it 99.81 percent of the time. A label that is 41 percent one way and 59 percent the other cannot be learned by any model, so keeping those rows would only teach the classifier contradictions. After cleaning, 19,813 reviews remained, of which 88.04 percent are positive.

That imbalance shapes the whole study. A model that reads nothing and answers "positive" every single time already scores 88.02 percent accuracy on our test set, so accuracy on its own is close to meaningless here. For this reason macro-F1 is used as the primary metric throughout, supported by the Matthews correlation coefficient and by recall on the negative class, which is the number a retailer actually cares about.

Ten classifiers were compared under identical conditions: logistic regression, linear support vector machine, multinomial naive Bayes, k-nearest neighbours, random forest, extra trees, XGBoost, LightGBM, CatBoost and a multi-layer perceptron. Two feature views were built from the same training text. Three models were given the full sparse TF-IDF matrix of 96,755 columns built from word unigrams, word bigrams and character n-grams. The remaining seven were given a dense 300-component truncated SVD projection of that same matrix. All ten models were tuned with Optuna against validation macro-F1 over a total of 184 trials, and each tuned model was scored exactly once on a held-out test set of 2,972 reviews. Two dummy classifiers were carried through the same evaluation code as controls.

Logistic regression finished first with 95.39 percent accuracy, macro-F1 0.8951, MCC 0.7914, ROC-AUC 0.9755 and 85.67 percent recall on the rare negative class, and it fits in 2.15 seconds. The linear SVM was second at 95.02 percent. All ten models beat both control baselines by a wide margin on macro-F1 and MCC. Three findings stand out. A simple tuned linear model beat three modern gradient boosting libraries and a neural network. The sparse feature view beat the dense one, because the 300-component projection keeps only 29.61 percent of the original variance and discards exactly the rare phrase features that decide a review. And accuracy without a stated baseline misleads badly, which is shown clearly by two models that score above 91 percent accuracy while missing more than half of all negative reviews.

**Keywords:** Sentiment Classification, E-Commerce Reviews, Class Imbalance, Macro-F1, Matthews Correlation Coefficient, TF-IDF, Truncated SVD, Logistic Regression, Gradient Boosting, Optuna, McNemar Test

---

\newpage

## TABLE OF CONTENTS

| Sl. No | Topic | Page |
|-------|-------|------|
| | Certificate | |
| | Declaration | |
| | Acknowledgements | |
| | Abstract | |
| | List of Figures | |
| | List of Tables | |
| | List of Abbreviations | |
| **1** | **Chapter 1: Introduction** | |
| | 1.1 Machine Learning | |
| | 1.2 Natural Language Processing and Text Classification | |
| | 1.3 Sentiment Analysis in E-Commerce | |
| | 1.4 The Problem with Online Clothing Reviews | |
| | 1.5 Problem Statement | |
| | 1.6 Objectives of the Project | |
| | 1.7 Scope of the Project | |
| | 1.8 Organisation of the Report | |
| **2** | **Chapter 2: Basic Concepts** | |
| | 2.1 The Study of Machine Learning | |
| | 2.1.1 Supervised Learning | |
| | 2.1.2 Unsupervised Learning | |
| | 2.1.3 Reinforcement Learning | |
| | 2.1.4 Regression | |
| | 2.1.5 Classification | |
| | 2.2 Theory | |
| | 2.2.1 Turning Text into Numbers | |
| | 2.2.2 TF-IDF Weighting | |
| | 2.2.3 Word n-grams and Character n-grams | |
| | 2.2.4 Sparse Matrices and Why They Matter | |
| | 2.2.5 Dimensionality Reduction with Truncated SVD | |
| | 2.2.6 Linear Decision Boundaries | |
| | 2.2.7 Probabilistic Classification | |
| | 2.2.8 Instance-Based Learning | |
| | 2.2.9 Bagging and Tree Ensembles | |
| | 2.2.10 Gradient Boosting | |
| | 2.2.11 Feed-Forward Neural Networks | |
| | 2.2.12 Class Imbalance and Class Weighting | |
| | 2.2.13 Hyperparameter Optimisation | |
| | 2.2.14 Statistical Testing Between Models | |
| **3** | **Chapter 3: Literature Review** | |
| | 3.1 Sentiment Analysis of Product Reviews | |
| | 3.2 Studies on Clothing and Fashion Review Data | |
| | 3.3 Transformer and Large Language Model Approaches | |
| | 3.4 Classical Machine Learning on E-Commerce Data | |
| | 3.5 Feature Representation for Short Text | |
| | 3.6 Class Imbalance and Choice of Metric | |
| | 3.7 Summary Table of Reviewed Literature | |
| | 3.8 Research Gap Addressed by This Work | |
| **4** | **Chapter 4: Project Analysis** | |
| | 4.1 Analysis of the Problem | |
| | 4.2 Why This Dataset Was Chosen | |
| | 4.3 Requirement Analysis | |
| | 4.4 Hardware and Software Used | |
| | 4.5 Feasibility Study | |
| | 4.6 System Workflow | |
| | 4.7 Design Decisions and Their Justification | |
| **5** | **Chapter 5: Implementation** | |
| | 5.1 Dataset Used | |
| | 5.2 Exploratory Data Analysis | |
| | 5.3 Data Preprocessing | |
| | 5.4 Data Splitting | |
| | 5.5 Feature Engineering | |
| | 5.6 Model Workflow | |
| | 5.7 The Ten Models | |
| | 5.8 Hyperparameter Tuning with Optuna | |
| | 5.9 Evaluation Metrics | |
| | 5.10 Control Baselines | |
| | 5.11 Software Structure of the Project | |
| **6** | **Chapter 6: Results and Discussion** | |
| | 6.1 The Baselines | |
| | 6.2 Main Model Comparison | |
| | 6.3 Per-Class Behaviour and the Accuracy Trap | |
| | 6.4 Ranking Curves and Threshold Behaviour | |
| | 6.5 Effect of Hyperparameter Tuning | |
| | 6.6 Computational Cost | |
| | 6.7 Statistical Significance | |
| | 6.8 What the Winning Model Learned | |
| | 6.9 Discussion | |
| | 6.10 Limitations | |
| **7** | **Chapter 7: Conclusion and Future Work** | |
| | 7.1 Conclusion | |
| | 7.2 Future Work | |
| **8** | **Chapter 8: Bibliography** | |
| **9** | **Chapter 9: Individual Contribution** | |

---

\newpage

## LIST OF FIGURES

| Sl. No | Title | Section |
|-------|-------|---------|
| 1 | Machine Learning Life Cycle | 1.1 |
| 2 | Types of Machine Learning | 2.1 |
| 3 | Regression and Classification Compared | 2.1.5 |
| 4 | Applications of Sentiment Analysis in E-Commerce | 1.3 |
| 5 | Converting a Review into a TF-IDF Vector | 2.2.2 |
| 6 | Word n-grams and Character n-grams from One Sentence | 2.2.3 |
| 7 | Truncated SVD Reducing 96,755 Columns to 300 | 2.2.5 |
| 8 | Complete Project Workflow | 4.6 |
| 9 | Rating Distribution Across All 23,486 Reviews | 5.2.1 |
| 10 | Recommendation Rate at Each Star Level | 5.2.2 |
| 11 | Class Balance After Cleaning | 5.2.3 |
| 12 | Review Length by Sentiment | 5.2.4 |
| 13 | Reviewer Age and Positive Rate by Department | 5.2.5 |
| 14 | Correlation Matrix of the Numeric Columns | 5.2.6 |
| 15 | Word Clouds for Positive and Negative Reviews | 5.2.7 |
| 16 | Most Discriminative Terms by Log-Odds Ratio | 5.2.8 |
| 17 | Cumulative Explained Variance of the SVD Projection | 5.5.3 |
| 18 | Accuracy and Macro-F1 by Model | 6.2 |
| 19 | Confusion Matrices for All Ten Models | 6.3 |
| 20 | ROC and Precision-Recall Curves | 6.4 |
| 21 | Optuna Search History for All Ten Models | 6.5 |
| 22 | Macro-F1 Against Fit Time | 6.6 |
| 23 | Holm-Corrected McNemar p-values for All 45 Pairs | 6.7 |
| 24 | The Twenty Strongest Terms for Each Class | 6.8 |

---

## LIST OF TABLES

| Sl. No | Title | Section |
|-------|-------|---------|
| I | Dataset Columns and Their Role in the Study | 5.1 |
| II | Rating Distribution in the Raw File | 5.2.1 |
| III | Data Quality Summary of the Raw File | 5.1 |
| IV | Cleaning Log | 5.3 |
| V | Recommendation Rate by Star Rating | 5.2.2 |
| VI | Class Balance After Cleaning | 5.2.3 |
| VII | Review Length Statistics by Class | 5.2.4 |
| VIII | Reviewer Age Statistics by Class | 5.2.5 |
| IX | Positive Rate by Division | 5.2.5 |
| X | Positive Rate by Department | 5.2.5 |
| XI | Positive Rate by Product Class | 5.2.5 |
| XII | Correlation Matrix of the Numeric Columns | 5.2.6 |
| XIII | Twenty Most Negative Terms by Log-Odds Ratio | 5.2.8 |
| XIV | Twenty Most Positive Terms by Log-Odds Ratio | 5.2.8 |
| XV | Stratified 70/15/15 Split | 5.4 |
| XVI | The Two Feature Representations | 5.5 |
| XVII | Cumulative Explained Variance at Selected Components | 5.5.3 |
| XVIII | The Ten Models, Their Families and Feature Views | 5.7 |
| XIX | Hardware and Software Environment | 4.4 |
| XX | Optuna Search Spaces and Trial Budgets | 5.8 |
| XXI | Selected Hyperparameters for Each Model | 5.8.3 |
| XXII | Control Baselines on the Test Set | 6.1 |
| XXIII | Main Comparison, Test Set, Ranked by Macro-F1 | 6.2 |
| XXIV | Per-Class Results on the Test Set | 6.3 |
| XXV | Confusion Matrix Counts for All Ten Models | 6.3 |
| XXVI | ROC-AUC and PR-AUC by Model | 6.4 |
| XXVII | Effect of Tuning, Ranked by Macro-F1 Gain | 6.5 |
| XXVIII | Tuning Cost and Best Validation Score | 6.5 |
| XXIX | Fit Time and Feature View | 6.6 |
| XXX | Significantly Different Model Pairs | 6.7 |
| XXXI | Model Pairs That Are Not Separable | 6.7 |
| XXXII | Twenty Strongest Negative Coefficients | 6.8 |
| XXXIII | Twenty Strongest Positive Coefficients | 6.8 |
| XXXIV | Deployment Recommendation by Scenario | 6.9.4 |
| XXXV | Objectives Set Against Outcomes Achieved | 7.1 |

---

## LIST OF ABBREVIATIONS

| Abbreviation | Full Form |
|---|---|
| AUC | Area Under the Curve |
| CPU | Central Processing Unit |
| CSV | Comma Separated Values |
| EDA | Exploratory Data Analysis |
| F1 | Harmonic Mean of Precision and Recall |
| GPU | Graphics Processing Unit |
| IDF | Inverse Document Frequency |
| k-NN | k-Nearest Neighbours |
| LLM | Large Language Model |
| LSTM | Long Short-Term Memory |
| MCC | Matthews Correlation Coefficient |
| ML | Machine Learning |
| MLP | Multi-Layer Perceptron |
| NB | Naive Bayes |
| NLP | Natural Language Processing |
| PCA | Principal Component Analysis |
| PR-AUC | Area Under the Precision-Recall Curve |
| RAM | Random Access Memory |
| ROC | Receiver Operating Characteristic |
| SMOTE | Synthetic Minority Over-sampling Technique |
| SVD | Singular Value Decomposition |
| SVM | Support Vector Machine |
| TF | Term Frequency |
| TF-IDF | Term Frequency - Inverse Document Frequency |
| TPE | Tree-structured Parzen Estimator |

---

\newpage
# Chapter 1
# Introduction

## 1.1 Machine Learning

Machine learning is the branch of artificial intelligence where a computer learns from examples instead of being given a rule for every case. Rather than writing instructions that cover every possible input, we collect a large number of past examples, show them to the computer, and let it work out the pattern on its own. The idea sounds small but it has changed how problems are solved in almost every industry, from hospitals reading scans to banks catching fraud to factories checking products on a line.

The machine learning life cycle begins with data collection. Companies gather data from cameras, sensors, transaction logs, forms, and in our case from what customers type into a review box. This raw data is the material everything else is built from. If the collection stage is careless, nothing later in the pipeline can fix it, so the quality of this first step decides the ceiling of the whole project.

Once the data is collected it has to be cleaned and prepared. Missing values must be handled, obviously broken rows must be dropped, and everything has to be brought into one consistent format. This step is often the longest part of a real project and it is where most of the errors hide. In our project the cleaning stage removed 845 rows with no review text, removed 2,823 reviews that carried an ambiguous label, and removed 5 exact duplicates. That is more than 3,600 rows out of 23,486, and the decision to remove each group had to be argued for rather than assumed.

After the data is ready, a model has to be chosen. The choice depends entirely on the shape of the problem. If the answer is a number, a regression model is needed. If the answer is one of a fixed set of labels, a classification model is needed. If the input is an image, convolutional networks are the usual starting point. If the input is a sentence, the choice is between counting words and feeding them into a linear model, or using a large pretrained language model. Available computing power matters as much as accuracy here, because a model that needs a graphics card for a week is not an option for a team that has a laptop and a deadline.

Training the model means showing it the prepared data so that it can adjust its internal settings until its predictions on that data are as close to the truth as it can manage. Depending on the model, this takes anywhere from a fraction of a second to several days. In our project the fastest model trained in 0.06 seconds and the slowest in 41.12 seconds, so training itself was never the bottleneck. The bottleneck was hyperparameter search, which took about 45 minutes for all ten models together.

After training comes tuning. Almost every model has settings that are not learned from the data and must be picked by the person building it, such as how strongly to penalise large weights, how deep a tree may grow, or how many neighbours to consult. These are called hyperparameters. Picking them by hand is slow and usually wrong, so an automatic search is used. In this project we used Optuna, which builds a small statistical model of which settings have worked well and samples the next attempt from the promising region rather than from a fixed grid.

Finally the trained and tuned model is tested on data it has never seen. This is the only number that means anything, because a model can memorise the data it was trained on without learning anything general. We held out 2,972 reviews, never touched them during training or tuning, and scored every model on them exactly once at the very end.

The growth of machine learning over the past fifteen years comes from three things happening together. There is far more data than there used to be, because almost every interaction is now recorded. Hardware got much faster and much cheaper. And the algorithms themselves improved, with gradient boosting libraries and transformer networks both appearing within the last ten years. What is easy to forget in that story is that the older, simpler methods did not stop working. One of the main findings of this project is that a method published in the nineteenth century in its statistical form, and standard in machine learning since the seventies, beat three libraries released after 2016.

*[Figure 1: Machine Learning Life Cycle]*

---

## 1.2 Natural Language Processing and Text Classification

Natural language processing is the part of machine learning that works with human language. It covers translation, question answering, summarising, speech recognition and many other tasks, but the oldest and most widely deployed application is text classification: reading a piece of writing and putting it into one of a small number of boxes. Spam filters do this. So do systems that route support tickets to the right department, systems that flag abusive comments, and systems that decide whether a news article is about sport or politics.

Text is difficult for a computer because a computer works with numbers and text is not numbers. The whole field rests on a conversion step that turns a sentence into a row of numbers a model can handle. The simplest version of this counts how many times each word in a fixed vocabulary appears in the document. A slightly better version weights those counts so that words appearing in almost every document count for very little, which is the idea behind TF-IDF. A more modern version passes the sentence through a pretrained neural network and uses the resulting vector. All three approaches are still in use, and the right choice depends on how much data there is, how much hardware is available, and how much accuracy is actually needed.

There is a real trade-off here and it is worth stating plainly at the start of this report. A fine-tuned transformer model will usually score one to three points higher than a well-built TF-IDF pipeline on review sentiment. It will also need a graphics card, take hours instead of seconds to train, and produce a model whose decisions cannot be read off directly. For a shop that wants to retrain its classifier every night on the reviews that arrived that day, the seconds matter more than the last two points. This project deliberately works within the cheaper approach and reports honestly what it costs.

---

## 1.3 Sentiment Analysis in E-Commerce

Sentiment analysis is text classification where the boxes are opinions. In the simplest form the boxes are positive and negative. Sometimes a neutral box is added, and sometimes the task is expanded to specific aspects, so that a single review can be positive about delivery and negative about sizing at the same time.

Online retail is the largest single consumer of this technology. Every product page on every large store carries reviews, and those reviews are the closest thing a retailer has to a direct line into what customers actually think. The numbers alone are not enough. A four-star average tells a manager that things are broadly fine. It does not tell them that the last two hundred reviews of one particular dress all complain that the zip breaks. That information is sitting in the text, and the only way to get at it at scale is to have a machine sort the text first.

The typical deployment looks like this. New reviews arrive continuously and are scored by the model as they come in. The share of negative reviews for each product is tracked over time. When that share jumps above a threshold, an alert is raised and the flagged reviews are sent to a human, who reads a few dozen instead of a few thousand. The value of the system is not that it understands clothing. It is that it sorts the pile so a person can spend their attention where it matters.

There are other places the same model is useful. Plenty of customer text arrives with no rating attached at all: support emails, live chat transcripts, social media mentions, comments under a product photo, and the free-text box on a returns form. A model trained on reviews, where the rating gives us a label for free, can then be pointed at all that unlabelled text. This is the main practical answer to the obvious objection that predicting a star rating is circular when the star rating is already in the database.

There is a second answer to that objection. The rating and the text do not always agree, and where they disagree the text is usually the more informative of the two. A customer who gives four stars and then writes a paragraph about the seams coming apart is telling the store something the number hides. A system that reads the text can find those cases; a system that reads the number cannot.

*[Figure 4: Applications of Sentiment Analysis in E-Commerce]*

---

## 1.4 The Problem with Online Clothing Reviews

Clothing is a harder case than most product categories, and that is what makes it interesting.

When someone buys a phone online, the important properties are written down. Screen size, battery capacity, storage, weight. A buyer can compare two phones on a specification table without reading a single review. Clothing has almost none of that. A dress has a fit, a feel, a way it hangs on a particular body, and a colour that may or may not match the photograph. None of those things are in the product data, so they end up in the review text instead. That makes written reviews far more important for clothing than for most other categories, and it makes the text richer and more varied.

It also makes the text harder to classify. Clothing reviews are full of qualifications. "Fitted but not tight" is praise. "Wanted to love it" is a complaint. "Runs small" is a warning rather than a judgement, and whether it is positive or negative depends on what the reviewer says next. A system that treats the review as an unordered bag of individual words will get many of these wrong, because the meaning lives in the pairs of words rather than in the words themselves.

On top of that, review data is heavily unbalanced. People who are happy with a purchase are more willing to write about it than people who are unhappy, partly because an unhappy customer usually goes through the returns process instead of the review box. In our dataset, 88.04 percent of usable reviews are positive after cleaning. This is not a footnote. It changes what every reported number means, and the whole design of this study follows from it.

The specific danger is that accuracy stops being informative. A model that reads nothing, learns nothing and answers "positive" every single time scores 88.02 percent accuracy on our test set. That number looks like a working system, and it is not one. It finds zero negative reviews, which is the entire reason the retailer wanted the system in the first place. Any report that says "we achieved 88 percent accuracy" without printing the baseline next to it has said nothing at all. We will print the baseline next to every table.

---

## 1.5 Problem Statement

Given a public dataset of 23,486 written reviews of women's clothing, each with a star rating from one to five, build and compare a set of machine learning classifiers that can decide from the review text alone whether the customer was satisfied or dissatisfied, and evaluate them using metrics that remain honest when one class outnumbers the other by more than seven to one.

The problem has four parts. The first is deciding what the label should be, given that a five-point rating scale has an ambiguous middle. The second is turning free-form English text into numbers without losing the phrase-level information that carries the meaning. The third is choosing and tuning a set of classifiers that covers the main families of machine learning rather than a single favourite. The fourth, and the one that most published work handles badly, is reporting the results in a way that a reader can trust, which means stating a baseline, ranking on a metric that respects the minority class, and checking whether the gaps between models are larger than random variation.

---

## 1.6 Objectives of the Project

The project set out with seven objectives.

**Objective 1.** To profile the raw dataset completely, including its schema, missing values, duplicate rows and rating distribution, and to record every cleaning action taken with the number of rows it removed.

**Objective 2.** To test, rather than assume, whether three-star reviews should be included in a two-class sentiment problem, using the recommendation flag in the dataset as independent evidence.

**Objective 3.** To build two different numeric representations of the same review text, one sparse and high dimensional and one dense and low dimensional, so that the effect of the representation can be separated from the effect of the model.

**Objective 4.** To train and compare ten classifiers drawn from five different families of machine learning, under identical data splits and identical evaluation code, so that the comparison is fair.

**Objective 5.** To tune every model with the same automatic search procedure against the same validation metric, with per-model trial budgets chosen so that no single expensive model consumes the whole computing budget.

**Objective 6.** To evaluate using macro-F1, Matthews correlation coefficient, Cohen's kappa, ROC-AUC, PR-AUC, per-class precision and recall, and accuracy, with two control baselines carried through the same code, and to lead every ranking with a metric that respects the minority class.

**Objective 7.** To test whether the differences between the ten models are statistically real, using McNemar's test on all 45 model pairs with a Holm-Bonferroni correction for multiple comparisons, and to report honestly which comparisons the data does not support.

---

## 1.7 Scope of the Project

The scope is deliberately narrow in some places and deliberately wide in others, and it is worth setting both out clearly.

The study is wide in its model coverage. Ten classifiers are compared, drawn from linear models, probabilistic models, instance-based methods, bagged tree ensembles, gradient boosting machines and neural networks. Every one of them is tuned rather than run at defaults, because comparing a tuned model against an untuned one is not a comparison at all. Two dummy classifiers are carried through the whole pipeline as controls, and every metric is reported against them.

The study is also wide in its evaluation. Seven different metrics are computed for every model, per-class results are reported alongside the aggregate ones, confusion matrix counts are printed in full, and all 45 pairwise comparisons are tested for significance.

The study is narrow in three places. First, the problem is two-class. Three-star reviews are removed, so the system is validated only on reviews that express a clear opinion. Second, no transformer model is included. Fine-tuning DistilBERT or DeBERTa needs a graphics card, and the machine used for this project has none, so a single fine-tuning run would have taken between six and ten hours. That belongs in future work and is stated as such. Third, the data comes from one retailer, in one country, in one product category. The vocabulary that separates the classes here is specific to clothing, and words such as "itchy" or "runs small" would carry no weight at all on electronics reviews.

---

## 1.8 Organisation of the Report

The report is arranged in nine chapters.

Chapter 1, the present chapter, introduces machine learning, natural language processing and sentiment analysis, sets out the problem this project solves, and lists the objectives and the scope.

Chapter 2 covers the basic concepts needed to follow the rest of the report. It explains the types of machine learning, the difference between regression and classification, and then works through the specific theory this project uses: TF-IDF, n-grams, sparse matrices, truncated SVD, linear boundaries, naive Bayes, nearest neighbours, bagging, boosting, neural networks, class weighting, hyperparameter search and significance testing.

Chapter 3 reviews the published literature on review sentiment analysis, covering classical machine learning studies, work specifically on clothing and fashion reviews, and recent transformer and large language model approaches, and ends by stating the gap this project addresses.

Chapter 4 analyses the project itself: why this problem, why this dataset, what the requirements were, what hardware and software were used, whether the plan was feasible, and what the main design decisions were.

Chapter 5 describes the implementation in full, from the raw file through exploratory analysis, cleaning, splitting, feature construction, the ten model implementations, the tuning procedure and the evaluation metrics.

Chapter 6 reports and discusses all results, including the baselines, the main comparison, per-class behaviour, the effect of tuning, computational cost, statistical significance and an interpretation of what the winning model actually learned.

Chapter 7 concludes and lists future work. Chapter 8 is the bibliography. Chapter 9 records the individual contribution of each team member.

---

\newpage
# Chapter 2
# Basic Concepts

The idea of a machine that learns from data goes back to the late nineteen fifties, when Arthur Samuel wrote a checkers program that improved by playing against itself and coined the phrase "machine learning" for what it was doing. The field has grown enormously since then, but the central idea has not changed. A computer is shown a collection of past examples, it finds patterns in them, and it uses those patterns to make a decision about a new example it has not seen before.

The standard process is simple to state. A model is fitted to a set of training data, absorbing whatever regularity is present in it. It is then tested on data it has never encountered, to see whether what it absorbed generalises or was just memorisation. That cycle is repeated with different settings until the tested performance is good enough for the job at hand.

This chapter covers the concepts a reader needs in order to follow the rest of the report. Section 2.1 covers machine learning in general. Section 2.2 covers the specific theory used in this project.

---

## 2.1 The Study of Machine Learning

Machine learning is usually divided into three broad types depending on what kind of feedback the learning algorithm gets.

*[Figure 2: Types of Machine Learning]*

### 2.1.1 Supervised Learning

In supervised learning every training example comes with the correct answer attached. The algorithm sees an input and the output it was supposed to produce, measures how wrong it was, and adjusts itself to be less wrong next time. Almost all deployed machine learning is supervised, because the presence of a correct answer makes the problem far easier and the results far easier to check.

This project is entirely supervised. Each review comes with a star rating, and that rating is turned into the correct answer the model has to predict. This is a common and convenient arrangement in review analysis, because the label is generated for free by the customer at the moment they write the review. Nobody has to sit down and label twenty thousand documents by hand.

### 2.1.2 Unsupervised Learning

In unsupervised learning there are no correct answers. The algorithm is given a pile of data and asked to find structure in it. Clustering algorithms group similar items together. Topic models find themes running through a collection of documents. Dimensionality reduction methods find a smaller set of directions that capture most of the variation in a wide dataset.

This project uses one unsupervised technique inside an otherwise supervised pipeline. Truncated singular value decomposition, described in Section 2.2.5, is fitted without ever looking at the labels. It finds the 300 directions along which the reviews differ most, and those directions are then handed to a supervised classifier. This kind of mixing is common and useful, because the unsupervised step can compress the data without knowing anything about the task.

### 2.1.3 Reinforcement Learning

In reinforcement learning an agent takes actions in an environment and receives rewards or penalties, and it learns a policy that maximises reward over time. There is no correct answer for any single action, only a score that arrives later. This is how programs learn to play games and how robots learn to move. It is not used anywhere in this project and is mentioned only for completeness.

### 2.1.4 Regression

Supervised learning splits into two families depending on what the correct answer looks like. If the answer is a number on a continuous scale, the task is regression. Predicting the price of a house, the temperature tomorrow, or the number of units a shop will sell next month are all regression problems.

Regression models are scored by how far their predictions land from the truth. Mean absolute error averages the size of the mistakes. Mean squared error averages the squares of the mistakes, which punishes large mistakes much more heavily than small ones. Root mean squared error takes the square root of that so the number is back in the original units. The coefficient of determination, usually written R squared, reports what share of the variation in the target the model managed to explain.

Regression is not used in this project, but it is worth mentioning because a five-star rating could in principle be treated as a number to be predicted rather than a class to be chosen. We decided against that. The gaps between rating levels are not equal in meaning, the distance from one star to two is not the same kind of thing as the distance from four to five, and a model that predicts 3.7 stars gives an operator nothing they can act on. A two-class decision is both easier to learn and easier to use.

### 2.1.5 Classification

If the correct answer is one of a fixed set of labels, the task is classification. Deciding whether an email is spam, which of ten digits a handwritten image shows, or whether a review is positive or negative are all classification problems.

Binary classification, where there are exactly two labels, is the simplest and most common case, and it is what this project does. The model outputs a score for each review, usually a probability between zero and one, and a threshold turns that score into a decision. The default threshold is 0.5, but that default is not sacred, and on imbalanced data moving it is often the cheapest way to trade one kind of error against another.

The four possible outcomes of a binary decision are worth naming carefully, because the rest of this report depends on them. A true positive is a positive review the model correctly called positive. A true negative is a negative review correctly called negative. A false positive is a negative review the model wrongly called positive. A false negative is a positive review wrongly called negative. In our setting the label "positive" belongs to the majority class, so the errors that hurt are the negative reviews called positive, and those are counted in the true negative and false positive cells rather than where a reader might first look. Chapter 6 prints all four counts for every model so there is no ambiguity.

*[Figure 3: Regression and Classification Compared]*

---

## 2.2 Theory

This section covers the specific techniques used in the project, in roughly the order they appear in the pipeline.

### 2.2.1 Turning Text into Numbers

Every classifier used in this project expects a row of numbers as input. A review is a string of characters. Bridging that gap is the single most important design decision in a text classification project, and it is made before any model is chosen.

The oldest and still the most widely used approach is the bag of words. A vocabulary is fixed, usually by taking every word that appears at least a couple of times in the training data. Each document is then represented by a row with one column per vocabulary word, holding the number of times that word appeared. The order of the words is thrown away entirely, which is why it is called a bag. "The dress is not lovely" and "The lovely dress is not" produce exactly the same row.

Losing word order sounds fatal and often is not, because for many tasks the presence of certain words is enough. A document containing "refund", "broken" and "disappointed" is almost certainly a complaint no matter what order those words appear in. But for review text the loss does bite in specific places, and Section 2.2.3 explains the standard repair.

### 2.2.2 TF-IDF Weighting

Raw counts have an obvious problem. The word "the" appears in nearly every review, so its count is large and completely uninformative. The word "itchy" appears in a handful of reviews and is almost perfectly predictive when it does. A raw count treats them as comparable numbers, which they are not.

TF-IDF fixes this by multiplying two quantities. Term frequency measures how often a token appears in the current document. Inverse document frequency measures how rare the token is across the whole collection, computed as the logarithm of the total number of documents divided by the number of documents containing that token. Multiplying the two gives a large value only when a token appears often in this document and rarely in the rest, which is exactly the definition of an informative word.

This project uses sublinear term frequency, which replaces the raw count with one plus its natural logarithm. The reason is that a word appearing ten times in a review is more important than a word appearing once, but it is not ten times more important. A customer who writes "cheap cheap cheap" is not making a point three times stronger than one who writes "cheap" once. The logarithm compresses that scale sensibly.

The vectoriser is fitted on the training data only. This matters more than it sounds. The vocabulary and the document frequencies are both learned quantities, and learning them from the full dataset would let information about the test reviews leak into the training stage. The fitted vectoriser is then applied unchanged to the validation and test text, and any word that appears only in those sets is simply dropped, which is the correct behaviour because a deployed model would have no column for a word it had never seen.

*[Figure 5: Converting a Review into a TF-IDF Vector]*

### 2.2.3 Word n-grams and Character n-grams

An n-gram is a sequence of n consecutive units. A word bigram is two consecutive words; a character trigram is three consecutive characters.

Adding word bigrams to the vocabulary is the standard repair for the lost word order described in Section 2.2.1. Instead of only having columns for "not" and "flattering", the matrix also gets a column for the pair "not flattering". This matters enormously in review text, where meaning is routinely carried by pairs whose parts are harmless on their own. "Not flattering" is negative while both of its words are neutral. "Runs small" is a warning while both of its words are ordinary. Section 6.8 of this report shows that the single most negative feature the winning model learned is the bigram "to love", which appears in sentences of the form "I wanted to love this dress, but the fabric was thin". No unigram model can represent that.

Character n-grams work at the level of letters rather than words. A character trigram vectoriser over the phrase "love it" produces "lov", "ove", "ve ", "e i", " it" and so on. This looks strange until you consider what review text actually looks like. People write on phones, quickly, and they misspell things. They write "loooove" and "gorgeus" and "recieved". A word-level vocabulary has no column for any of those, so the information is lost. A character-level vocabulary catches them, because "loooove" still contains "lov" and "ove". Character features also pick up word endings and short function patterns that survive across spelling variants.

This project uses both. The word vectoriser covers unigrams and bigrams. The character vectoriser uses the `char_wb` mode, which builds character n-grams of length three to five but keeps them inside word boundaries rather than running across the spaces between words. The outputs of the two vectorisers are joined side by side into one wide matrix.

*[Figure 6: Word n-grams and Character n-grams from One Sentence]*

### 2.2.4 Sparse Matrices and Why They Matter

The combined matrix in this project has 96,755 columns and 13,869 training rows. Stored as ordinary numbers that would be over 1.3 billion values, which at eight bytes each is more than ten gigabytes. That will not fit in the memory of an ordinary laptop.

It does not need to. Almost every entry is zero. A review of sixty words can touch at most a few hundred of the 96,755 columns, so 99.9 percent of the matrix is empty. A sparse matrix stores only the positions and values of the non-zero entries and treats everything else as implicitly zero. The same data then takes a few tens of megabytes instead of ten gigabytes.

Sparse storage is not just a memory trick, because it changes which algorithms are usable. Linear models can be trained on a sparse matrix efficiently, since the update for each row only touches the handful of columns that row actually uses. Decision trees cannot, because a tree has to consider splitting on every column at every node, and with 96,755 columns and the vast majority of them almost entirely zero that search is both slow and useless. Distance-based methods cannot either, because in a space that wide and that empty almost every pair of points ends up at a similar distance from every other pair, and the notion of a nearest neighbour stops carrying information.

This is the technical reason the project has two feature views rather than one, and it is not an arbitrary choice.

### 2.2.5 Dimensionality Reduction with Truncated SVD

Singular value decomposition factors a matrix into three parts, and the middle part lists the singular values in decreasing order of importance. Keeping only the largest few and discarding the rest gives the best possible approximation of the original matrix at that reduced size. Truncated SVD does exactly this, computing only the components that are kept rather than the full decomposition, which is what makes it practical on a matrix this wide.

Applied to a document-term matrix the technique is often called latent semantic analysis. The intuition is that the 300 retained directions correspond loosely to themes running through the collection, and each document is described by how strongly it expresses each theme instead of by which exact words it contains. Documents using different words for the same idea end up close together, which is a genuine advantage over raw counts.

Truncated SVD is preferred over ordinary principal component analysis here for a practical reason. PCA centres the data by subtracting the mean of each column, and subtracting a non-zero mean from a sparse matrix fills in all the zeros and destroys the sparsity. Truncated SVD works on the uncentred matrix directly and keeps the sparsity intact.

The cost of the reduction is information, and in this project that cost is measured rather than assumed. The 300 components retain 29.61 percent of the original variance, so roughly seven tenths of the detail is thrown away. Chapter 6 shows this loss appearing in the results as a one to three point accuracy penalty for every model that uses the reduced view. The reason is specific: a rare but highly predictive word contributes very little to the total variance of the matrix, so no early component preserves it, and the model never sees it.

*[Figure 7: Truncated SVD Reducing 96,755 Columns to 300]*

### 2.2.6 Linear Decision Boundaries

A linear classifier gives every input feature a weight, multiplies each feature value by its weight, adds the results together with a constant offset, and decides the class from the sign of the total. Geometrically this draws a flat surface through the feature space and assigns points to classes depending on which side they fall.

Logistic regression is the most common linear classifier. It passes the weighted sum through the logistic function, which squashes any number onto the range zero to one, and reads that as the probability of the positive class. Training adjusts the weights to make the predicted probabilities match the observed labels as closely as possible, with a penalty term discouraging weights from growing too large. The strength of that penalty is controlled by a parameter usually written C, where a small C means heavy penalty and a simple model and a large C means light penalty and a model free to fit the data closely.

The linear support vector machine draws a flat boundary too but picks it differently. Instead of fitting probabilities it looks for the boundary that leaves the widest empty margin between the two classes. Only the points closest to the boundary affect where it goes, so distant points cannot drag it around. That tends to make it steadier than logistic regression when the data contains outliers. The base implementation returns a distance from the boundary rather than a probability, and probabilities are needed for ROC and precision-recall curves, so the classifier has to be wrapped in a calibration step that converts distances into probabilities using internal cross-validation.

Linear models are often dismissed as too simple for serious problems. On very wide, sparse text data that dismissal is wrong, and understanding why is central to this report. When there are far more features than training rows, the two classes can almost always be separated by some flat surface. Fitting a curved boundary in that situation does not help, because there was nothing curved to find; it only gives the model more ways to memorise noise. A linear model is the correct assumption for that shape of data, not a weaker one.

### 2.2.7 Probabilistic Classification

Naive Bayes takes a different route to the same decision. It uses Bayes' theorem to ask which class was more likely to have produced the document that was observed, given how often each token appears in each class in the training data.

The word "naive" refers to its central assumption: that every token is independent of every other token given the class. This is obviously false for language, since the presence of "not" changes the meaning of whatever follows it. The model works well anyway, and the usual explanation is that the false assumption mainly damages the calibration of the probabilities rather than the ranking of the classes. The model may say a document is 99.99 percent positive when the honest figure is 80 percent, but it still puts the right class on top, and for classification that is what matters.

The multinomial version used here models token counts and has one important hyperparameter, the additive smoothing constant alpha. Its job is to handle tokens that never appeared with a particular class in training. Without smoothing, such a token would give that class a probability of exactly zero and would veto the class no matter what the rest of the document said. Alpha adds a small pseudo-count to every token so that nothing is impossible, only unlikely.

Naive Bayes is included in this project as the classical text-classification baseline and because it is extremely fast. It fits in 0.06 seconds, which is roughly four hundred times faster than the slowest boosting model in the study.

### 2.2.8 Instance-Based Learning

k-nearest neighbours has no training phase at all. It stores the training data. When a new document arrives it finds the k stored documents closest to it under some distance measure, and takes a vote among their labels.

The method is called instance-based or lazy because all the work happens at prediction time. It makes no assumption whatsoever about the shape of the boundary between the classes, which is its main theoretical attraction: if the true boundary is complicated in a way no parametric model captures, nearest neighbours can still follow it.

It has two well-known weaknesses that both apply here. The first is the curse of dimensionality. As the number of dimensions grows, the distances between points become more and more similar to each other, and the idea of a nearest neighbour becomes less and less meaningful. This is why k-NN in this project runs on the 300-component projection rather than the 96,755-column sparse matrix. The second weakness is that it inherits the class balance of its neighbourhoods. In a dataset where 88 percent of points are positive, most neighbourhoods are mostly positive, so the vote leans positive almost everywhere. Chapter 6 shows this happening: k-NN reaches 91.18 percent accuracy while finding only 156 of the 356 negative reviews in the test set.

It is included in this study deliberately, as a contrast. A method that fails on this data tells us something about the data.

### 2.2.9 Bagging and Tree Ensembles

A decision tree splits the data repeatedly on one feature at a time, choosing at each step the split that best separates the classes, until the leaves are pure enough or a depth limit is reached. A single tree is easy to read and easy to overfit. Grown deep enough, a tree can memorise the training data completely and generalise badly.

Bagging fixes this by averaging. Many trees are grown, each on a different random sample of the rows drawn with replacement, and their votes are averaged. The individual trees still overfit, but they overfit differently, so the errors partly cancel. A random forest adds a second source of randomness on top: at every split, only a random subset of the features is considered as candidates. This decorrelates the trees further, which is what makes the averaging effective.

Extremely randomised trees, usually called extra trees, push the same idea one step further. Instead of searching for the best threshold on each candidate feature, they draw a threshold at random and keep the best of those random draws. Each tree is weaker as a result, but they are faster to grow and less correlated with each other, and sometimes the ensemble comes out ahead.

Both methods are strong general-purpose classifiers on ordinary tabular data. Both are used in this project on the 300-component dense view, because neither can work on the sparse matrix.

### 2.2.10 Gradient Boosting

Boosting builds trees in sequence rather than in parallel. The first tree is fitted to the data. The second tree is fitted to the errors the first one made. The third is fitted to what the first two still get wrong, and so on. Each tree is small, often only three or four levels deep, and the final prediction is the sum of all of them with a learning rate controlling how much each contributes.

The approach is powerful because it concentrates effort where the model is currently failing rather than spreading it evenly. It is also easy to overfit, which is why the learning rate is kept small, the trees are kept shallow, and the number of trees is chosen carefully.

Three boosting libraries are compared in this project and they differ in real ways.

XGBoost uses second-order gradient information, which means it uses the curvature of the loss function as well as its slope when deciding splits, and it adds explicit penalties on the leaf weights. It grows trees level by level and offers a histogram-based split finder that bins continuous features to speed up training.

LightGBM grows trees leaf by leaf instead, always splitting whichever leaf promises the largest improvement wherever it sits in the tree. This produces deeper, more unbalanced trees for the same number of splits and usually trains several times faster than level-wise growth. It bins features into histograms before training as a standard step rather than an option.

CatBoost builds symmetric trees, meaning every node at the same depth uses the same split condition. This acts as a strong regulariser and makes prediction very fast, since evaluating a symmetric tree reduces to computing an index. It also uses ordered boosting, a scheme designed to remove the subtle target leakage that ordinary boosting introduces when the same rows are used both to compute gradients and to fit the tree that corrects them.

All three are strong. All three lose to logistic regression in this study, and Chapter 6 explains why.

### 2.2.11 Feed-Forward Neural Networks

A multi-layer perceptron is the simplest kind of neural network. Input values are multiplied by weights and summed at each unit of a hidden layer, a non-linear activation function is applied to each sum, and the results are passed on to the next layer until the output layer produces a class probability. Training uses backpropagation, which computes how much each weight contributed to the error and adjusts it accordingly, repeated over many passes through the data.

The non-linear activation is what separates this from a linear model. With it, the network can represent curved decision boundaries, and with enough hidden units it can approximate any reasonable function. That flexibility is also its risk: with more parameters than the data can constrain, it will fit noise. Regularisation through an L2 penalty on the weights and early stopping based on a validation score are the usual defences, and both are used here.

The network in this project is small by modern standards: a single hidden layer of 128 units taking the 300 SVD components as input. It is included specifically to test whether the boundary between positive and negative reviews is curved. The answer, given in Chapter 6, is that it does not appear to be, since the network finishes behind both linear models.

### 2.2.12 Class Imbalance and Class Weighting

When one class heavily outnumbers the other, most learning algorithms drift towards the majority, because predicting the majority everywhere is a cheap way to reduce the average error. On an 88 to 12 split this drift is severe, and several models in this study collapsed almost entirely onto the majority class at their default settings.

There are three standard responses.

The first is resampling. Either the majority class is thinned out, or the minority class is duplicated, or synthetic minority examples are generated by interpolating between existing ones, which is what SMOTE does. Resampling changes the data itself.

The second is class weighting. The loss function is modified so that a mistake on a minority example counts for more than a mistake on a majority one, usually in inverse proportion to how common the classes are. The data is untouched; only the accounting changes. Every model in this study that offered a class weighting option had it included in its search space.

The third is threshold adjustment. The model is trained as usual, but the cut-off that turns a probability into a decision is moved away from 0.5 to a value chosen on validation data.

This project relies on class weighting, and Chapter 6 reports a clear finding about it. Across all ten models, class weighting turned out to be the single most useful thing the hyperparameter search could change, more useful than any learning rate or tree depth. Extra trees gained 0.3327 macro-F1 from tuning and almost all of that came from switching the class weighting on.

### 2.2.13 Hyperparameter Optimisation

Hyperparameters are the settings a model does not learn from data: the regularisation strength, the number of trees, the depth limit, the learning rate, the number of neighbours. They have to be chosen from outside, and the choice usually matters more than which model was picked in the first place.

Grid search tries every combination on a fixed grid. It is exhaustive and simple and it becomes impossible very quickly, because the number of combinations multiplies with each parameter added. Random search samples combinations at random, which sounds worse but is usually better in practice, because most parameters do not matter much and random sampling covers the ones that do more efficiently than a grid.

Bayesian optimisation is smarter than both. It keeps a model of which regions of the parameter space have produced good scores, and it chooses each new trial from where good scores look likely. Optuna, the library used here, implements a tree-structured Parzen estimator, which maintains two probability distributions over parameter values, one fitted to the trials that scored well and one to the trials that scored badly, and samples where the ratio between them is most favourable.

Two design points about the search in this project are worth stating early. First, every study optimises validation macro-F1, never accuracy. Optimising accuracy on an 88 to 12 split would reward a model for ignoring the minority class, which is the opposite of what is wanted. Second, the trial budget is set per model rather than flat, because a naive Bayes fit takes 0.06 seconds while a CatBoost fit takes 26 seconds. A flat budget would spend nearly the whole runtime inside the two most expensive models and starve the rest.

### 2.2.14 Statistical Testing Between Models

If two models score 95.39 percent and 95.02 percent on the same test set, is the first one better? Not necessarily. On 2,972 test reviews a gap of 0.37 percentage points is eleven reviews. Eleven reviews is well within the range that a different random split would move.

McNemar's test answers this properly for two classifiers evaluated on the same data. It ignores the cases where both models agree, since those carry no information about which is better, and looks only at the disagreements: the number of cases where model A was right and model B wrong, and the number where the reverse happened. If the two models are equally good, those two counts should be roughly equal, and the test measures how far from equal they actually are.

Running the test on many pairs raises a second problem. With ten models there are 45 pairs, and testing 45 hypotheses at the five percent level means that even if every model were identical, we would expect about two pairs to look significant by chance alone. The Holm-Bonferroni procedure corrects for this. It sorts the p-values from smallest to largest and applies a progressively less severe correction to each, which controls the chance of any false positive across the whole family of tests while being less conservative than simply multiplying every p-value by 45.

This project runs McNemar's test on all 45 pairs and applies the Holm correction to all of them. The result, reported in Section 6.7, is that 25 pairs differ significantly and 20 do not, and one of the pairs that does not differ significantly is the gap between the first and second placed models. Reporting that honestly is part of the point of this study.

---

\newpage
# Chapter 3
# Literature Review

Sentiment classification of product reviews is a well-studied problem, and the published work on it splits into a few clear strands. This chapter goes through those strands one at a time, starting with the general problem, moving to work done specifically on clothing and fashion data, then to the recent transformer and large language model results, and finally to the parts of the literature that deal with feature representation and with evaluation under class imbalance. The chapter ends with a summary table and a statement of the gap this project addresses.

## 3.1 Sentiment Analysis of Product Reviews

Deciding whether a piece of writing is positive or negative is one of the oldest applied tasks in natural language processing. The work that set the pattern for everything after it was the study by Pang, Lee and Vaithyanathan on film reviews [17], which established the basic recipe still in use today: break the text into small units, count them, weight the counts, and fit a linear classifier. That paper also made the point, unusual at the time, that human intuitions about which words signal sentiment are worse than the words a machine picks out from data.

Two later datasets shaped the field more than any single method. Maas and colleagues released a large collection of polarised film reviews and trained word vectors on it [18]. Socher and colleagues released a treebank with sentiment labelled at every level of the parse tree, which allowed models to be tested on their handling of composition and negation rather than only on overall polarity [19]. Both of those datasets removed the middle of the rating scale when constructing their binary versions, and this is worth noting because it is precisely the decision this project makes and justifies with measurements in Chapter 5.

The field then moved through word embeddings to transformer models, which now hold the best published scores on every standard benchmark. Those gains come at a cost in hardware. A fine-tuned transformer needs a graphics card and hours of training, while the approach used in this project trains in seconds on an ordinary laptop processor. For a retailer that wants to retrain its classifier weekly on a normal server, that difference matters more than the last two points of accuracy.

## 3.2 Studies on Clothing and Fashion Review Data

Several studies work on the same kind of data as this project.

**Sentiment analysis of women's clothing reviews [1]:** Mahmud and colleagues applied a machine learning pipeline to women's clothing reviews on e-commerce platforms, treating the task as polarity classification from the review body. The study establishes the basic viability of classical classifiers on this kind of text and is the closest published match to the dataset used here.

**SVM, ANN and BERT on women's e-commerce clothing reviews [2]:** Yabes and colleagues compared a support vector machine, an artificial neural network and BERT on the task of classifying and predicting rating sentiment from women's e-commerce clothing reviews. The comparison is useful because it places a transformer next to two classical methods on the same data, which is exactly the comparison this project could not afford to run on its own hardware.

**Advanced machine learning on e-commerce clothing reviews [3]:** Prince and colleagues applied a set of advanced machine learning algorithms to e-commerce clothing review analysis. The study reports strong accuracy figures across several classifiers but does not report a majority baseline alongside them, which makes the reported numbers difficult to place. This is a recurring pattern in the applied literature and is one of the things this project sets out to do differently.

**Aspect-based sentiment analysis of Vietnamese clothing reviews [4]:** Quoc-Hung and colleagues extended the problem from overall polarity to aspect-level sentiment on Vietnamese clothing reviews, so that a single review can be positive about one property of a garment and negative about another. This is a more useful output for a retailer than a single polarity label, and it is one of the natural extensions of the present work.

**Fashion styles from large review collections [5]:** Hananto and colleagues used machine learning to analyse fashion styles from large collections of online customer reviews. Their target is not polarity at all but style categorisation, and the study is included here because it shows that the same review text supports several different commercially useful predictions, not only sentiment.

## 3.3 Transformer and Large Language Model Approaches

The most recent work on review sentiment is dominated by pretrained language models.

**IndoBERT against XLM-RoBERTa [6]:** Putra and Lhaksmana compared IndoBERT and XLM-RoBERTa for sentiment classification of Indonesian e-commerce product reviews from a human-centred perspective. Both models performed well, but XLM-RoBERTa generalised better on the held-out test set, reaching F1 scores of 0.973 on the positive class and 0.974 on the negative class against 0.956 and 0.957 for IndoBERT.

**OnDeBERTa [7]:** Liu and colleagues proposed an ordered-neuron memory enhanced DeBERTa model for e-commerce review sentiment classification and compared it against pretrained language models, decoder-augmented variants and conventional machine learning under one experimental procedure on the Amazon Review Polarity dataset. The one-layer OnDeBERTa outperformed DeBERTa-base, DeBERTa with LSTM and DeBERTa with BiLSTM, reaching 94.18 percent accuracy, 92.26 percent precision, 91.02 percent recall and an F1 score of 91.64 percent.

**Fine-tuned large language models for Turkish reviews [8]:** Kiziltepe and colleagues studied sentiment analysis for a low-resource language using fine-tuned large language models on Turkish customer reviews. Their fine-tuned models performed six percent better than leading commercial large language models on both in-domain and out-of-domain tests. Models fine-tuned on labels generated by GPT performed best, with in-domain and out-of-domain F1 scores of 0.912 and 0.9184.

**Lexicon methods against RoBERTa [10]:** Kumar, Badiger and Jacintha compared a lexicon-based approach using VADER against RoBERTa for product review sentiment. RoBERTa reached roughly 91 percent accuracy and clearly beat the lexicon method, which is the expected result but is useful as a calibration point for how far a rule-based system can get without learning.

**VADER, BERT and RoBERTa compared [11]:** Adarsh, Mohla and Mahto examined sentiment classification of product reviews across VADER, BERT and RoBERTa. Their reported F1 scores were 68.9 for BERT, 57.8 for RoBERTa and 12.7 for VADER. The very low lexicon score is a useful warning that a rule-based sentiment dictionary transfers badly to specific product domains, where the vocabulary that separates the classes is not general English sentiment vocabulary at all.

**Deep sequence models on product reviews [12]:** Atlas and colleagues applied BiGRU and RNN-based LSTM deep learning models to product review sentiment analysis. Sequence models of this kind sit between bag-of-words methods and transformers in both cost and capability, since they read the words in order but do not carry the large pretrained knowledge a transformer brings.

## 3.4 Classical Machine Learning on E-Commerce Data

A separate strand of work applies classical classifiers to e-commerce problems that are not strictly sentiment.

**Customer satisfaction prediction on Brazilian e-commerce [13]:** Pandey and colleagues predicted customer satisfaction on the Brazilian Olist dataset and compared six methods. Their LSTM reached 87.09 percent, followed by XGBoost at 85.36 percent, naive Bayes at 84.64 percent, a neural network at 83.97 percent, random forest at 83.34 percent and logistic regression at 81.22 percent. This is one of the two studies that supplied the original comparison points for our own work.

**Demand forecasting on the same source data [14]:** Debbagh and colleagues worked on the Olist data but treated the problem as demand forecasting rather than classification, reporting a mean squared error of 0.0045 and a root mean squared error of 0.0067. Neither this study nor the previous one removed a neutral class, and neither reported a majority baseline next to its accuracy figures.

**Logistic regression against random forest for customer behaviour [15]:** Sunarya and colleagues compared logistic regression with random forest for predicting e-commerce customer behaviour. Studies of this kind are relevant here because they test the same two model families on similar data and reach conclusions that vary with the dataset, which supports the view that no single family is universally best.

**Naive Bayes variants for text classification [16]:** Kalcheva, Marinova and Todorova compared the Bernoulli and multinomial naive Bayes classifiers for text classification. The multinomial variant, which models token counts rather than only presence and absence, is the one used in this project, and their comparison supports that choice for documents of review length.

**Fake review detection [9]:** Khan, Adnan and Anwar used sentiment analysis together with linguistic features such as review length, punctuation, misspellings, adjective usage and textual similarity to detect fraudulent reviews. Decision tree and random forest classifiers reached strong F1 scores and up to 98 percent accuracy across their datasets. This work is relevant to ours as a caution: any review dataset may contain fabricated reviews, and a sentiment model trained on them learns the sentiment of the fabrication rather than of a real customer.

## 3.5 Feature Representation for Short Text

TF-IDF remains the default representation for classical text classifiers, and the weighting scheme itself dates back to the work of Salton and Buckley on term weighting in automatic text retrieval [20].

Two refinements matter for review data specifically. The first is the inclusion of word bigrams, which capture short phrases whose meaning is not the sum of their parts. "Not flattering" is the clearest example, since both words are harmless on their own. The second is character n-grams, which survive spelling mistakes, informal spellings and word endings that a word-level vocabulary would miss. Reviews are written quickly on phones and contain plenty of both. This project uses word and character features together for exactly these reasons, and Section 6.8 shows the effect appearing directly in the learned coefficients.

On the reduction side, the randomised algorithms described by Halko, Martinsson and Tropp [21] are what make truncated SVD practical on a matrix of nearly a hundred thousand columns. Without them the decomposition would be too slow to sit inside a project of this size.

## 3.6 Class Imbalance and the Choice of Metric

When one class dominates, accuracy tracks the majority class and hides failure on the minority. The literature on this is settled, and this project follows it closely.

Macro-averaged F1 computes the F1 score separately for each class and averages them with equal weight regardless of class size, which is exactly the correction an imbalanced problem needs. The Matthews correlation coefficient goes further by summarising all four cells of the confusion matrix in one number that only rises when the model does well on both classes at once. Chicco and Jurman argue in detail that MCC is the most honest single figure for imbalanced binary classification and that F1 and accuracy can both be misleading [28]. We follow that advice, and we also print the majority baseline beside every table.

On the testing side, McNemar's test [29] is the standard procedure for comparing two classifiers evaluated on the same test set, and Dietterich's analysis of approximate statistical tests for comparing supervised learning algorithms [31] recommends it specifically for this situation. The Holm-Bonferroni procedure [30] controls the family-wise error rate when many pairs are tested at once, without being as conservative as a plain Bonferroni correction.

## 3.7 Summary Table of Reviewed Literature

Table A summarises the studies most directly comparable to this work.

**TABLE A. SUMMARY OF REVIEWED LITERATURE**

| Ref | Year | Primary author | Domain | Best model | Reported result | Baseline stated |
|---|---|---|---|---|---|---|
| [6] | 2026 | Putra A. A. | Indonesian e-commerce reviews | XLM-RoBERTa | 0.973 F1 | No |
| [7] | 2026 | Liu J. | Amazon review polarity | OnDeBERTa | 94.18% accuracy | No |
| [8] | 2025 | Kiziltepe R. S. | Turkish customer reviews | Fine-tuned LLaMA-3.2-1B-IT | 0.912 F1 | No |
| [9] | 2025 | Khan C. | Fake review detection | Random Forest | 98% accuracy | No |
| [12] | 2025 | Atlas L. G. | Product reviews | BiGRU / LSTM | Not directly comparable | No |
| [10] | 2024 | Kumar B. | Product reviews | RoBERTa | 91% accuracy | No |
| [11] | 2024 | Adarsh V. | Product reviews | BERT | 68.9 F1 | No |
| [3] | 2024 | Prince N. U. | E-commerce clothing | Advanced ML ensemble | High accuracy | No |
| [4] | 2024 | Quoc-Hung P. | Vietnamese clothing | Aspect-based model | Aspect-level F1 | No |
| [15] | 2024 | Sunarya P. A. | E-commerce behaviour | Random Forest | Comparative study | No |
| [13] | 2023 | Pandey T. N. | Brazilian Olist | LSTM | 87.09% accuracy | No |
| [1] | 2023 | Mahmud F. A. M. | Women's clothing reviews | Classical ML pipeline | Not stated here | No |
| [2] | 2023 | Yabes I. | Women's clothing reviews | BERT | Comparative study | No |
| [5] | 2021 | Hananto V. R. | Fashion style analysis | Classical ML | Style categorisation | No |
| - | 2026 | **This work** | **Women's clothing reviews** | **Logistic Regression** | **95.39% accuracy, 0.8951 macro-F1** | **Yes, 88.02%** |

The last column is the one worth looking at. Of the fourteen studies summarised above, none reports the majority-class baseline of its dataset next to its headline accuracy. That is not a criticism of any individual paper; it is a habit of the field. But it makes the numbers hard to compare with each other and hard to compare with ours, because an accuracy figure only means something once the reader knows what a model that learned nothing would have scored.

## 3.8 Research Gap Addressed by This Work

Published comparisons of many classifiers on review data tend to share four weaknesses, and this project was designed around fixing them.

**They report accuracy first.** On imbalanced review data this is close to meaningless, and this project leads every table with macro-F1 and MCC instead, with accuracy reported last and always beside the baseline.

**They keep the ambiguous middle of the rating scale, or drop it without testing whether that helps.** This project measures the ambiguity directly using the recommendation flag, finds that only 41.41 percent of three-star reviewers recommended the product, and reports that number as the justification for removal rather than citing convention.

**They do not test whether the gaps between models are larger than random variation.** This project runs McNemar's test on all 45 model pairs with a Holm-Bonferroni correction, and reports openly that the difference between the first and second placed models is not statistically significant.

**They confound the model with its input representation.** A study that gives sparse features to one model and dense features to another and then ranks the models has measured the pair, not the model. This project has the same confound by design, because seven of the ten models cannot use the sparse matrix, but it states this plainly, measures the variance lost by the projection at 29.61 percent, and treats the resulting gap as a representation effect rather than a claim about model quality.

To this we add a fifth point that is less about method and more about honesty of framing. Most published results in this area are presented as if the winning model were clearly the best. In our study the top five models are not separable at the size of test set available, and saying so is more useful to a reader than declaring a winner the data does not support.

---

\newpage
# Chapter 4
# Project Analysis

## 4.1 Analysis of the Problem

The problem this project solves is easy to state and harder to do properly. A retailer has a large pile of written reviews and wants to know which of them are complaints. The obvious approach is to train a classifier on the review text using the star rating as the label. Almost every step after that decision hides a trap.

The first trap is the label itself. A five-point scale does not divide cleanly into two classes. One and two stars are clearly negative and four and five are clearly positive, but three stars sits in the middle and could go either way. Most published work either keeps three-star reviews and forces them into one class, or drops them silently. Both choices affect the results, and neither is usually defended. Our analysis of this problem was to look for independent evidence inside the dataset itself. The dataset records, separately from the rating, whether the customer would recommend the product. That gives us a way to ask how consistent each rating level actually is, and the answer settles the question. One-star reviewers refuse to recommend 98.10 percent of the time. Five-star reviewers recommend 99.81 percent of the time. Three-star reviewers split 58.59 percent against 41.41 percent, which is nearly a coin toss. A three-star review is not a weak signal; it is a mixed one, and training a two-class model on mixed signals teaches it contradictions.

The second trap is the class balance. After cleaning, 88.04 percent of reviews are positive. This is a normal ratio for review data and it is not a defect to be corrected, because a deployed system will meet the same ratio. But it destroys accuracy as a metric. A model that always answers "positive" scores 88.02 percent on our test set, and a reader who sees "90 percent accuracy" without a baseline cannot tell whether that model learned anything. Our analysis of this problem was to fix the metric hierarchy before running any model, so that no result could be presented in a flattering way after the fact. Macro-F1 leads. MCC checks it. Negative-class recall gives the business meaning. Accuracy is reported last, next to the baseline, every time.

The third trap is the representation. Text has to become numbers, and the way it becomes numbers determines what the model can possibly learn. A bag of individual words cannot represent "not flattering" as a unit. A dense projection cannot preserve a rare word that appears in forty reviews out of fourteen thousand. Our analysis of this problem was to build both a wide sparse representation and a narrow dense one from the same training text, give each model whichever it can actually use, and measure how much information the reduction costs so that the resulting gap can be explained rather than guessed at.

The fourth trap is the comparison itself. Comparing a carefully tuned favourite model against a set of untuned alternatives is not a comparison. Our analysis of this problem was to tune every model with the same optimiser against the same validation metric, and to allocate trial budgets per model in proportion to cost so that the expensive models were not starved and the cheap ones did not soak up the whole budget.

## 4.2 Why This Dataset Was Chosen

The Women's E-Commerce Clothing Reviews collection was chosen over other publicly available review datasets for five reasons.

It is real commercial data rather than a research construction. The reviews were written by actual customers of an actual United States retailer and were anonymised before release, so no customer or brand can be identified. The messiness in it is real messiness: missing titles, missing bodies, duplicated rows, wildly uneven ratings. A dataset that has been cleaned before publication teaches a student nothing about cleaning.

It carries a second, independent signal alongside the rating. The recommendation flag is what allowed the neutral-class question in Section 4.1 to be answered with a measurement instead of an assumption. Most review datasets have a rating and nothing else, so the same analysis would not have been possible.

It is the right size. At 23,486 rows it is large enough that the results are stable and a held-out test set of nearly three thousand reviews is available, and small enough that ten models can be tuned and evaluated on a laptop in about an hour. A dataset ten times larger would have forced us to cut either the number of models or the tuning budget.

It has an interesting imbalance. At roughly 88 to 12 the minority class is rare enough that naive approaches fail visibly, but not so rare that the problem becomes anomaly detection. That is the range where the choice of metric matters most, and metric choice is one of the things this project is about.

It is in a category where the text carries the information. As set out in Section 1.4, clothing quality is not describable in a specification table, so customers describe it in sentences. That makes the review text genuinely load-bearing rather than decorative.

## 4.3 Requirement Analysis

**Functional requirements.** The system must load the raw dataset from a fixed source, profile it and record its schema, missing values and duplicates. It must apply a documented sequence of cleaning steps and log the number of rows removed at each one. It must construct a binary target from the star rating. It must split the cleaned data into training, validation and test sets in a stratified and reproducible way. It must build two numeric representations of the review text, fitted on the training split only. It must train ten classifiers at default settings and again at tuned settings. It must compute a fixed set of metrics for every model on the test set. It must run pairwise significance tests across all model pairs. It must export every table to a spreadsheet and every chart to an image file at publication resolution.

**Non-functional requirements.** The whole pipeline must run on a laptop CPU with no graphics card. The end-to-end run must complete in about an hour so that experiments can be repeated. Every random operation must be seeded so that a second run reproduces the first exactly. The code must be organised so that a single change to a setting propagates everywhere it is used, rather than being repeated in several files. No test data may influence any fitting step at any point.

**Data requirements.** The dataset must contain review text, a rating, and enough rows that a fifteen percent test split still contains a few hundred minority examples. The 2,972-row test set used here contains 356 negative reviews, which is enough to compute stable per-class metrics but is small enough to limit the power of the significance tests, and Chapter 6 says so explicitly.

## 4.4 Hardware and Software Used

Everything in this project was run on a single ordinary laptop. No cloud computing, no graphics card and no distributed setup were used at any point. This was a constraint rather than a choice, and it shaped the study: it is the reason no transformer model appears in the comparison, and it is the reason the trial budgets in Section 5.8 are set per model rather than flat.

**TABLE XIX. HARDWARE AND SOFTWARE ENVIRONMENT**

| Component | Specification |
|---|---|
| Processor | Intel Core i7-1255U, 12th generation, 1.70 GHz base |
| Memory | 16 GB RAM |
| Graphics | None used; entire study is CPU only |
| Operating system | Windows 11 |
| Language | Python 3 |
| Core libraries | scikit-learn, NumPy, pandas, SciPy |
| Boosting libraries | XGBoost, LightGBM, CatBoost |
| Tuning | Optuna |
| Statistics | statsmodels for McNemar's test |
| Plotting | Matplotlib, seaborn, wordcloud |
| Export | openpyxl for the result workbooks |
| Data source | kagglehub, for reproducible dataset download |
| Environment | Jupyter Notebook with supporting Python modules |

The measured stage timings on this machine were as follows. Loading, cleaning and the full exploratory analysis with all fourteen figures took about two minutes. Splitting and building both feature representations took about forty seconds. Training all ten models at default settings took about ninety seconds. The Optuna tuning stage across all ten models took 2,728.6 seconds, which is roughly forty-five minutes and is by far the largest single cost. Producing the final figures and writing the seven result workbooks took about a minute.

## 4.5 Feasibility Study

**Technical feasibility.** Every component of the pipeline is available as a mature open-source library, and none of them requires special hardware. The largest object in memory at any point is the sparse TF-IDF matrix, which at 13,869 rows by 96,755 columns with 99.9 percent zeros occupies a few tens of megabytes rather than the ten gigabytes a dense array would need. The tuning stage is the only expensive part, and it was brought within budget by allocating trials per model.

**Economic feasibility.** The total cost of the project was zero beyond a laptop that already existed. The dataset is public. Every library used is free and open source. No cloud compute was purchased. This matters beyond the student context, because it means the resulting system is also cheap for a small retailer to run: a model that fits in 2.15 seconds on a laptop can be retrained nightly on a modest server without anyone noticing the cost.

**Time feasibility.** The end-to-end pipeline runs in about an hour, which meant the whole experiment could be re-run whenever a design decision changed. This turned out to matter. The neutral-class analysis was run first, then the whole pipeline was run on data with the neutral class retained, and then again with it removed, so that the effect could be measured rather than argued about. A pipeline that took a day per run would have made that comparison impractical and the decision would have rested on convention instead of evidence.

**Operational feasibility.** The output of the system is a probability per review, which slots directly into the workflow described in Section 1.3: score incoming reviews, track the negative share per product, alert when it rises, and route the flagged reviews to a person. Nothing in that workflow requires the retailer to change how reviews are collected.

## 4.6 System Workflow

The workflow runs in one direction from the raw file to the final comparison. Figure 8 draws it in full.

The CSV is downloaded through kagglehub so that no manual upload is involved and the source is fixed and reproducible. The raw file is profiled first: schema, data types, missing counts, duplicate counts and the rating distribution are all recorded before anything is changed.

The cleaning steps described in Section 5.3 are then applied in a fixed order, and the row count after each step is logged so that the effect of each decision is visible. The cleaned data is split into training, validation and test sets, and that split is frozen for the rest of the study. Every model sees exactly the same rows in exactly the same roles.

Two feature views are built from the training text only. The first is a sparse TF-IDF matrix combining word n-grams of length one and two with character n-grams of length three to five, giving 96,755 columns. The second is a dense matrix produced by projecting that same TF-IDF matrix down to 300 components with truncated SVD and then standardising the result so all 300 columns share a scale. The vectorisers, the SVD and the scaler are all fitted on the training split alone and then applied unchanged to validation and test, so no information from the evaluation data reaches the training stage.

Each of the ten models is then trained twice: once with sensible default settings, and once with hyperparameters chosen by Optuna against validation macro-F1. Only the tuned models are scored on the test set, and they are scored exactly once.

Finally the ten models are compared with each other using McNemar's test on their test predictions, the resulting p-values are corrected with the Holm-Bonferroni procedure, and the coefficients of the best linear model are extracted so that the model can be interpreted rather than only scored. Two dummy classifiers run through the same evaluation code from beginning to end, so that every metric in the report has a floor to sit on.

![Figure 8](paper_figures/fig00_workflow.png)

**Figure 8.** Complete project workflow. Three models consume the sparse TF-IDF matrix directly; the other seven consume a 300-component SVD projection of the same matrix.

## 4.7 Design Decisions and Their Justification

Seven decisions shaped this project more than any others. Each is stated here with the reason behind it, so that a reader can disagree with the reasoning rather than guess at it.

**Decision 1: remove three-star reviews.** The justification is the measured 41.41 percent recommendation rate among three-star reviewers, against 98.10 percent rejection at one star and 99.81 percent recommendation at five stars. Removing them cost 2,823 rows, which is 12.22 percent of the data, and raised the best accuracy from 90.93 percent to 95.56 percent in preliminary runs on identical models. The removal is reported as a scope restriction, not as an improvement: the resulting system is validated only on reviews that express a clear opinion.

**Decision 2: build the target from the rating rather than the recommendation flag.** Both were available. The rating was chosen because it is the more widely available signal across retailers and because it gives a graded scale from which the ambiguous middle can be identified and removed. The recommendation flag was kept aside and used as independent evidence for Decision 1, which would not have been possible if it had been the target.

**Decision 3: join the title to the review body.** Titles are missing in 3,810 rows, so a naive approach would either drop those rows or drop the title column entirely. Joining the two fields with a space, and substituting an empty string where the title is missing, keeps every usable row and lets short headlines such as "Runs small" contribute to the prediction.

**Decision 4: apply no stemming, lemmatisation or stop-word removal.** This runs against common practice and was deliberate. Sublinear TF-IDF weighting already suppresses common words, so stop-word removal adds nothing. More importantly, standard stop-word lists contain "not", "but" and "too", and Section 6.8 shows that three of the strongest features the winning model learned are "but not", "not too" and "to love". Removing those words would have removed the model's best evidence.

**Decision 5: build two feature views instead of one.** Seven of the ten models cannot use a 96,755-column sparse matrix for the reasons in Section 2.2.4. The alternative would have been to reduce every model to the dense view, which would have handicapped the linear models to make the comparison superficially fairer. We chose instead to give each model the representation it can actually use, to measure the variance lost by the reduction, and to state clearly in Chapter 6 that the ranking measures model plus representation rather than model alone.

**Decision 6: tune against macro-F1, never accuracy.** On an 88 to 12 split, optimising accuracy rewards a model for ignoring the minority class. Every Optuna study in this project optimises validation macro-F1, and Section 6.5 shows one case, CatBoost, where the search deliberately traded 0.0030 accuracy for 0.0193 macro-F1, which is exactly the behaviour the objective was chosen to produce.

**Decision 7: carry two dummy classifiers through the entire pipeline.** The majority baseline always predicts positive; the stratified baseline predicts at random in proportion to the training class rates. Both are evaluated by the same code that evaluates the real models, so every metric reported anywhere in this report has a stated floor. This is the single cheapest thing a study can do to keep its own results honest, and it is the thing the reviewed literature in Chapter 3 most consistently omits.

---

\newpage
# Chapter 5
# Implementation

## 5.1 Dataset Used

The data is the Women's E-Commerce Clothing Reviews collection, released publicly on Kaggle by the user `nicapotato` [34]. It holds 23,486 reviews written by real customers of a United States clothing retailer. The set has been anonymised so that neither the customers nor the brand can be identified, which is why the reviews refer to garments only in general terms.

There are ten columns. `Clothing ID` is an item number. `Age` is the reviewer's age in years. `Title` is the short headline a customer types above their review. `Review Text` is the review body. `Rating` is the star score from one to five. `Recommended IND` is a flag saying whether the customer would recommend the item. `Positive Feedback Count` records how many other shoppers marked the review as helpful. `Division Name`, `Department Name` and `Class Name` describe where the item sits in the store's catalogue.

Only two of those columns carry the signal this project uses. `Review Text` is the main input and `Title` is joined to it. `Rating` supplies the label. Everything else is context, and Section 5.2.6 shows through direct measurement that the context columns carry almost nothing.

**TABLE I. DATASET COLUMNS AND THEIR ROLE IN THE STUDY**

| Column | Type | Role in this study |
|---|---|---|
| Clothing ID | integer | Dropped. Keeping it risks item-level memorisation rather than sentiment learning |
| Age | integer | Context only; examined in the exploratory analysis |
| Title | text | **Input**, joined to the review body |
| Review Text | text | **Input**, the main signal |
| Rating | integer 1 to 5 | **Source of the target label** |
| Recommended IND | binary | Used as independent evidence for the neutral-class removal |
| Positive Feedback Count | integer | Context only; recorded after purchase, so unavailable at prediction time in practice |
| Division Name | category | Context only |
| Department Name | category | Context only |
| Class Name | category | Context only |

`Clothing ID` deserves a note, because dropping it was a decision rather than an oversight. It identifies the specific garment being reviewed, and some garments in the catalogue attract far more negative reviews than others. A model given that column could learn which item numbers are bad instead of learning what a complaint sounds like, and it would score well on this dataset while being useless on any new product. Since the purpose is a system that reads text, the column had to go.

`Positive Feedback Count` was dropped for a different reason. It records how many other shoppers found the review helpful, which is information that only exists after the review has been public for some time. A deployed system scores a review the moment it arrives, when that count is zero for every review. Using it would have inflated the reported numbers with information a live system will never have.

**TABLE III. DATA QUALITY SUMMARY OF THE RAW FILE**

| Property | Value |
|---|---|
| Total rows | 23,486 |
| Total columns | 10 |
| Rows with missing `Review Text` | 845 |
| Rows with missing `Title` | 3,810 |
| Fully duplicated rows | 21 |
| Rows with a rating of 3 | 2,871 |
| Distinct `Clothing ID` values | Over 1,000 |
| Rating range | 1 to 5, integer |
| Age range | 18 to 99 |

## 5.2 Exploratory Data Analysis

Before any model was built, the dataset was examined closely. This section reports what was found, because several of the design decisions in Chapter 4 follow directly from these measurements rather than from prior expectation.

### 5.2.1 Rating Distribution

The ratings lean heavily towards the top of the scale. More than half of all reviews give five stars, and the two negative levels together account for just over ten percent of the data.

**TABLE II. RATING DISTRIBUTION IN THE RAW FILE**

| Rating | Count | Share of all reviews |
|---|---|---|
| 1 star | 842 | 3.59% |
| 2 stars | 1,565 | 6.66% |
| 3 stars | 2,871 | 12.22% |
| 4 stars | 5,077 | 21.62% |
| 5 stars | 13,131 | 55.91% |
| **Total** | **23,486** | **100%** |

This shape is normal for review data and it comes from how people behave rather than from anything about the retailer. Customers who are satisfied are more willing to leave a review, and a customer who is unhappy enough to act usually starts a return rather than writing about it. The consequence for us is that the positive class will dominate no matter how the labels are defined, and every metric decision in this project follows from accepting that rather than trying to hide it.

![Figure 9](paper_figures/fig01_rating_distribution.png)

**Figure 9.** Rating distribution across all 23,486 reviews. Five-star reviews alone make up 55.91 percent of the data.

### 5.2.2 Evidence for Removing the Neutral Class

This is the most important measurement in the project. The dataset records the recommendation flag separately from the rating, which gives an independent way to ask how consistent each rating level is.

**TABLE V. RECOMMENDATION RATE BY STAR RATING**

| Rating | Did not recommend | Recommended | n | Share of data |
|---|---|---|---|---|
| 1 | 98.10% | 1.90% | 842 | 3.59% |
| 2 | 93.99% | 6.01% | 1,565 | 6.66% |
| **3** | **58.59%** | **41.41%** | **2,871** | **12.22%** |
| 4 | 3.31% | 96.69% | 5,077 | 21.62% |
| 5 | 0.19% | 99.81% | 13,131 | 55.91% |

Read the middle row against the rows above and below it. At one star, 98.10 percent of customers say they would not recommend the item, which is close to unanimous. At five stars, 99.81 percent say they would, which is even closer. At two and four stars the agreement is still strong at 93.99 percent and 96.69 percent. Then at three stars the pattern collapses to 58.59 against 41.41, which is barely distinguishable from a coin toss.

That is a measurement, not an opinion. Three-star reviewers do not share a common judgement about the product, so there is no consistent thing for a two-class model to learn from them. Whichever class they are assigned to, roughly two out of every five of them will carry a label that contradicts what the same customer said elsewhere in the same record. Keeping 2,823 such rows in a training set of fourteen thousand would actively damage the model.

Two supporting points make this a defensible methodological choice rather than a convenient one. First, it is standard practice: both the IMDB and SST-2 sentiment benchmarks construct their binary versions by discarding the middle of the scale. Second, we measured the effect rather than assuming it. On preliminary runs across the same set of models, keeping the neutral class capped the best achievable accuracy at 90.93 percent against a baseline of 81.89 percent, while removing it raised the best accuracy to 95.56 percent against a baseline of 88.04 percent.

The honest way to describe what the removal buys is this. It does not make the model better at reading English. It removes 12.22 percent of the data whose labels were close to random, and a label that is 41.41 percent one way cannot be learned by any model at all. The cost is that the resulting system is validated only on reviews that express a clear opinion, and Section 6.10 lists this as the first limitation of the study.

![Figure 10](paper_figures/fig02_neutral_justification.png)

**Figure 10.** Recommendation rate at each star level. One-star and five-star reviewers are almost unanimous. Three-star reviewers split near 40 to 60, which is why they are removed.

### 5.2.3 Class Balance After Cleaning

After the cleaning described in Section 5.3, 19,813 reviews remain and the binary target is built by mapping ratings of four and five to positive and ratings of one and two to negative.

**TABLE VI. CLASS BALANCE AFTER CLEANING**

| Class | Count | Share |
|---|---|---|
| Positive (rating 4 or 5) | 17,443 | 88.04% |
| Negative (rating 1 or 2) | 2,370 | 11.96% |
| **Total** | **19,813** | **100%** |

The ratio is roughly seven and a third positive reviews for every negative one. This single number is the reason the metric hierarchy in Section 5.9 is arranged the way it is, and it is worth carrying it in mind for the whole of Chapter 6.

![Figure 11](paper_figures/fig03_class_balance.png)

**Figure 11.** Class balance after cleaning. 17,443 reviews are positive at 88.04 percent and 2,370 are negative at 11.96 percent.

### 5.2.4 Review Length

A reasonable guess before looking at the data is that unhappy customers write longer reviews, since a complaint usually needs explaining while praise does not. We checked this, and the guess is wrong.

**TABLE VII. REVIEW LENGTH STATISTICS BY CLASS**

| Class | Mean words | Median words | Std. dev. words | Max words | Mean characters | Median characters | Max characters |
|---|---|---|---|---|---|---|---|
| Negative | 63.98 | 63 | 27.72 | 118 | 331.76 | 327 | 557 |
| Positive | 62.26 | 61 | 29.38 | 121 | 322.11 | 313 | 553 |

Negative reviews average 63.98 words against 62.26 for positive ones. That is a gap of 1.72 words, or under three percent, on distributions whose standard deviations are close to thirty words. The medians differ by two words. The maximum lengths are almost identical. Whatever separates the classes here, it is not how much the customer wrote.

This has a practical consequence. Review length was considered as an additional numeric feature and rejected on the basis of this table, because a feature that differs by three percent between classes while varying by fifty percent within them cannot contribute anything a classifier can use.

![Figure 12](paper_figures/fig04_review_length.png)

**Figure 12.** Review length by sentiment. The two distributions sit almost on top of each other.

### 5.2.5 Reviewer Age and Product Category

The same check was run on the reviewer's age and on the catalogue columns, with the same result.

**TABLE VIII. REVIEWER AGE STATISTICS BY CLASS**

| Class | Mean age | Median age | Std. dev. |
|---|---|---|---|
| Negative | 42.99 | 41 | 11.97 |
| Positive | 43.50 | 41 | 12.45 |

The two classes differ by half a year in mean age and not at all in median age. Age carries nothing.

**TABLE IX. POSITIVE RATE BY DIVISION**

| Division Name | Positive rate | n |
|---|---|---|
| General | 87.71% | 11,624 |
| General Petite | 88.20% | 6,907 |
| Intimates | 90.07% | 1,269 |

**TABLE X. POSITIVE RATE BY DEPARTMENT**

| Department Name | Positive rate | n |
|---|---|---|
| Bottoms | 90.26% | 3,254 |
| Intimate | 90.02% | 1,473 |
| Jackets | 88.16% | 912 |
| Tops | 87.47% | 8,747 |
| Dresses | 87.19% | 5,315 |
| Trend | 78.79% | 99 |

Across the five substantial departments the positive rate moves only between 87.19 percent for Dresses and 90.26 percent for Bottoms, a spread of three points. The Trend department at 78.79 percent looks different, but it holds only 99 reviews, which is too few to lean on. With that many rows the difference could easily be a handful of unlucky garments.

**TABLE XI. POSITIVE RATE BY PRODUCT CLASS**

| Class Name | Positive rate | n |
|---|---|---|
| Lounge | 92.20% | 590 |
| Layering | 91.87% | 123 |
| Jeans | 91.44% | 993 |
| Shorts | 90.57% | 265 |
| Pants | 89.80% | 1,196 |
| Skirts | 89.36% | 799 |
| Intimates | 89.31% | 131 |
| Fine gauge | 88.94% | 940 |
| Legwear | 88.73% | 142 |
| Jackets | 88.42% | 622 |
| Sleep | 88.38% | 198 |
| Knits | 87.62% | 4,021 |
| Outerwear | 87.59% | 290 |
| Sweaters | 87.28% | 1,187 |
| Dresses | 87.19% | 5,315 |
| Swim | 86.81% | 288 |
| Blouses | 86.80% | 2,599 |
| Trend | 78.79% | 99 |
| Casual bottoms | 100% | 1 |
| Chemises | 100% | 1 |

Ignoring the two classes with a single review each, the spread across product classes runs from 86.80 percent to 92.20 percent, which is under six points across categories that differ enormously in what they are. Nothing here would let a model tell a complaint from a compliment.

![Figure 13](paper_figures/fig05_age_and_department.png)

**Figure 13.** Reviewer age and positive rate by department. Neither separates the classes.

### 5.2.6 Correlation Between the Numeric Columns

The final check on the metadata was a straightforward correlation matrix between every numeric column and the target.

**TABLE XII. CORRELATION MATRIX OF THE NUMERIC COLUMNS**

| | Age | Feedback Count | Rating | Recommended | Sentiment | Review words |
|---|---|---|---|---|---|---|
| **Age** | 1.000 | 0.042 | 0.019 | 0.018 | 0.013 | 0.014 |
| **Positive Feedback Count** | 0.042 | 1.000 | −0.055 | −0.059 | −0.057 | 0.191 |
| **Rating** | 0.019 | −0.055 | 1.000 | 0.865 | 0.910 | −0.047 |
| **Recommended IND** | 0.018 | −0.059 | 0.865 | 1.000 | 0.930 | −0.026 |
| **Sentiment (target)** | 0.013 | −0.057 | 0.910 | 0.930 | 1.000 | −0.019 |
| **Review words** | 0.014 | 0.191 | −0.047 | −0.026 | −0.019 | 1.000 |

The outcome is one-sided and easy to read. The target correlates at 0.910 with `Rating` and at 0.930 with `Recommended IND`, but neither of those is a genuine predictor. The target was derived from the rating, so that correlation is the target correlating with itself, and the recommendation flag is the same customer judgement recorded a second time. The two of them correlate with each other at 0.865 for the same reason.

Every real feature is close to useless on its own. Age correlates with sentiment at 0.013. Positive feedback count correlates at −0.057. Review length in words correlates at −0.019. The only correlation among the genuine features that is not near zero is between review length and positive feedback count at 0.191, which says that longer reviews get marked helpful more often. That is believable and it has nothing to do with sentiment.

The conclusion drawn from Sections 5.2.4 through 5.2.6 is that the metadata carries almost no signal and the text carries all of it. That is why the models in this project are trained on text alone, and it is a finding rather than an assumption, because we checked before deciding.

![Figure 14](paper_figures/fig06_correlation_heatmap.png)

**Figure 14.** Correlation matrix of the numeric columns. The only strong values involve `Rating` and `Recommended IND`, which are restatements of the label.

### 5.2.7 Word Clouds

Word clouds were produced for each class as a first visual check on whether the text separates at all. They are not evidence in a strict sense, since a word cloud shows frequency rather than discriminative power, and words common to both classes will appear large in both. They are useful as a sanity check, and they confirmed that the negative cloud is dominated by words about fit, fabric and disappointment while the positive cloud is dominated by words about love, comfort and fit working out.

![Figure 15](paper_figures/fig07_wordclouds.png)

**Figure 15.** Word clouds for positive and negative reviews.

### 5.2.8 Most Discriminative Terms

A more useful measure than raw frequency is the log-odds ratio, which compares how often a term appears in one class against how often it appears in the other. A term that is common in both classes scores near zero no matter how frequent it is, and only terms that genuinely favour one class score highly.

**TABLE XIII. TWENTY MOST NEGATIVE TERMS BY LOG-ODDS RATIO**

| Term | Log-odds | Term | Log-odds |
|---|---|---|---|
| felt cheap | −5.078 | feels cheap | −3.722 |
| quality poor | −4.472 | poor | −3.720 |
| worst | −4.035 | unwearable | −3.646 |
| poor quality | −3.959 | disappointed really | −3.597 |
| cute model | −3.933 | huge disappointment | −3.597 |
| poorly | −3.913 | wanted like | −3.528 |
| horrible | −3.840 | ill fitting | −3.469 |
| poor execution | −3.779 | looks cheap | −3.469 |
| great idea | −3.779 | execution | −3.469 |
| awful | −3.765 | high hopes | −3.461 |

**TABLE XIV. TWENTY MOST POSITIVE TERMS BY LOG-ODDS RATIO**

| Term | Log-odds | Term | Log-odds |
|---|---|---|---|
| fits great | 3.497 | great summer | 3.003 |
| booties | 3.493 | definitely recommend | 2.954 |
| subtle | 3.291 | highly | 2.945 |
| super comfortable | 3.233 | keeper | 2.925 |
| highly recommend | 3.230 | cooler | 2.865 |
| dressed | 3.063 | fits perfectly | 2.808 |
| perfect fall | 3.023 | versatile | 2.741 |
| great shirt | 2.695 | xs fits | 2.713 |
| absolutely love | 2.647 | flats | 2.671 |
| compliments | 2.620 | little big | 2.618 |

Two things in these tables are worth pointing out before Chapter 6 returns to them.

The negative list contains three entries whose words are individually positive: "cute model", "great idea" and "high hopes". All three appear in the same review structure, where the customer sets up an expectation and then says it was not met. "This was such a great idea but the execution is poor" is the archetype, and the phrase "poor execution" sits four rows above "great idea" in the same table.

The positive list contains "little big", which reads like a complaint. In clothing reviews it usually is not, because it appears in sentences like "runs a little big so I sized down and it was perfect". The sizing advice is delivered as a mild negative and the overall verdict is positive.

Both patterns are invisible to a system that looks at single words, and both are exactly what the bigram features described in Section 5.5 exist to catch.

![Figure 16](paper_figures/fig08_discriminative_terms.png)

**Figure 16.** Most discriminative terms by log-odds ratio.

## 5.3 Data Preprocessing

Preprocessing runs in four steps and each one is recorded with the number of rows it removed.

**Step 1. Drop rows with an empty review body.** A text classifier has nothing to work with when there is no text, so the 845 rows with a missing `Review Text` were removed. Rows with a missing `Title` were kept, and the title was replaced with an empty string, because the body alone is still perfectly usable. This is why 845 rows go and 3,810 rows stay despite both being missing values.

**Step 2. Join the title to the body.** The title and the body are joined into one field with a single space between them. This lets short headlines such as "Runs small" or "Beautiful but sheer" contribute to the prediction instead of being thrown away. Titles are short but they are often the sharpest summary of the reviewer's opinion, since a customer writing a headline compresses their verdict into a few words.

**Step 3. Remove all three-star reviews.** All 2,823 reviews rated three stars in the remaining data were removed, for the reason measured in Section 5.2.2.

**Step 4. Remove exact duplicate texts.** Five reviews whose joined text is an exact duplicate of an earlier review were removed. This step matters more than the small number suggests. If a duplicated review were left in, the same sentence could land in both the training and the test set, and the model would score well on it for the wrong reason. Removing five rows costs nothing and closes that hole completely.

The binary target is then built by mapping ratings of four and five to positive and ratings of one and two to negative.

**TABLE IV. CLEANING LOG**

| Stage | Rows remaining | Rows removed | Reason |
|---|---|---|---|
| Raw file | 23,486 | 0 | Starting point |
| Empty review text removed | 22,641 | 845 | Nothing for a text classifier to read |
| Rating = 3 removed | 19,818 | 2,823 | Label is ambiguous; only 41.41% recommended |
| Duplicate texts removed | 19,813 | 5 | Prevents the same text appearing in train and test |

One further decision belongs in this section because of what it does not do. No stemming, no lemmatisation and no stop-word removal were applied. This is against common practice and it was deliberate. Sublinear TF-IDF weighting already suppresses common words, so stop-word removal adds nothing that the weighting has not already done. And standard English stop-word lists contain "not", "but", "too" and "very". Section 6.8 shows that three of the strongest features the winning model learned are the phrases "but not", "not too" and "to love". Applying a stop-word list would have deleted the model's best evidence before it ever saw the data.

## 5.4 Data Splitting

The 19,813 cleaned reviews are divided into three parts: 13,869 for training, 2,972 for validation and 2,972 for testing, which is a 70 / 15 / 15 split.

The split is stratified, meaning the share of positive reviews is held almost constant across the three parts. This matters because the negative class is small. An ordinary random split could easily hand one set several dozen more negative reviews than another, and every reported number would then shift from run to run for reasons that have nothing to do with the models. With stratification the positive rate comes out at 88.04 percent, 88.06 percent and 88.02 percent across the three sets, which is as close as integer counts allow.

A fixed random seed of 42 is used everywhere, so the split is reproducible. Re-running the pipeline produces the same rows in the same sets.

The three sets have separate jobs and were never mixed. The training set fits the models. The validation set chooses hyperparameters and does nothing else. The test set is touched exactly once, at the very end, to produce every number in Chapter 6. No model was ever retrained after seeing a test score, because doing so would turn the test set into a second validation set and the reported figures would become optimistic.

**TABLE XV. STRATIFIED 70/15/15 SPLIT**

| Split | n | Positive | Negative | Positive rate | Share of data |
|---|---|---|---|---|---|
| Train | 13,869 | 12,210 | 1,659 | 88.04% | 70% |
| Validation | 2,972 | 2,617 | 355 | 88.06% | 15% |
| Test | 2,972 | 2,616 | 356 | 88.02% | 15% |
| **Total** | **19,813** | **17,443** | **2,370** | **88.04%** | **100%** |

The number to carry forward from this table is 356. That is how many negative reviews exist in the test set, and every interesting error in Chapter 6 comes out of that count. When a model is described as finding 305 negatives, it means 305 out of 356.

## 5.5 Feature Engineering

### 5.5.1 The Sparse View

The sparse representation is built by two TF-IDF vectorisers whose outputs are joined side by side.

The word vectoriser uses unigrams and bigrams, ignores any term appearing in fewer than two documents, applies sublinear term frequency, and caps the vocabulary at 50,000 features. Bigrams are the important part of this configuration, for the reasons given in Section 2.2.3 and demonstrated in the discriminative-term tables above.

The character vectoriser uses `char_wb` mode with n-grams of length three to five, a minimum document frequency of three, sublinear term frequency, and the same 50,000-feature cap. The `char_wb` setting builds character n-grams within word boundaries rather than running across the spaces between words, which keeps the features meaningful. Character features handle the misspellings and stretched words that appear constantly in review text.

The two matrices are joined with a feature union, giving a combined matrix of 96,755 columns. It is 99.9 percent zeros and is held in sparse storage throughout.

Both vectorisers are fitted on the training split only and then applied unchanged to validation and test. Any term that appears only in the evaluation data is simply dropped, which is the correct behaviour, because a deployed model has no column for a word it has never seen.

### 5.5.2 The Dense View

Seven of the ten models cannot work on a 96,755-column matrix, for the reasons given in Section 2.2.4. For those models the same sparse matrix is projected down to 300 components with truncated SVD.

The SVD is fitted on the training matrix alone. The resulting 300-column matrix is then standardised with a scaler, also fitted on training only, so that all 300 columns have zero mean and unit variance. Standardisation matters here because the SVD components have very different scales, with the first components carrying much larger values than the last, and distance-based and gradient-based methods behave badly when input columns differ in scale by orders of magnitude.

**TABLE XVI. THE TWO FEATURE REPRESENTATIONS**

| Representation | Columns | Storage | Used by |
|---|---|---|---|
| TF-IDF, word 1-2 grams plus char_wb 3-5 grams | 96,755 | Sparse | Logistic Regression, Linear SVM, Multinomial NB |
| Truncated SVD, 300 components, standardised | 300 | Dense | k-NN, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, MLP |

### 5.5.3 What the Projection Costs

The reduction is not free and the cost was measured rather than assumed. The 300 retained components hold 29.61 percent of the variance in the original matrix, so roughly seven tenths of the detail is discarded.

**TABLE XVII. CUMULATIVE EXPLAINED VARIANCE AT SELECTED COMPONENTS**

| Components kept | Cumulative variance explained |
|---|---|
| 1 | 0.21% |
| 2 | 0.74% |
| 3 | 1.24% |
| 6 | 2.31% |
| 11 | 3.75% |
| 26 | 6.83% |
| 51 | 10.69% |
| 101 | 16.37% |
| 151 | 20.67% |
| 201 | 24.21% |
| 251 | 27.17% |
| 300 | 29.61% |

The shape of this curve is worth reading carefully. There is no elbow anywhere in it. The first fifty components pick up variance quickly, and after that the curve settles into a slow and almost straight climb: components 201 to 251 still add 2.96 percentage points between them, against 3.54 for components 151 to 201. Nothing suggests a natural cut-off point. That is the signature of data whose information is spread thinly across a very large number of features rather than concentrated in a few strong directions, which is exactly what text data looks like.

The practical consequence is that no reasonable number of components would have captured most of the variance. Reaching even fifty percent would have needed well over a thousand components, which would have removed most of the speed advantage that motivated the reduction in the first place. Three hundred was chosen as a compromise that keeps the tree models trainable in seconds, and Chapter 6 reports what that compromise costs in accuracy.

What gets lost is not random. A rare word that appears in forty reviews out of 13,869 contributes almost nothing to the total variance of the matrix, so no early component preserves it, even if it is nearly perfectly predictive when it does appear. "Itchy" is that kind of word. A linear model on the full sparse matrix simply gives it a large weight and uses it. The tree models never see it at all.

![Figure 17](paper_figures/fig15_svd_variance.png)

**Figure 17.** Cumulative explained variance of the truncated SVD projection. The curve has no elbow, which is the signature of information spread thinly across a very large number of features.

## 5.6 Model Workflow

With the data split and both representations built, the modelling stage proceeds in a fixed order.

Each of the ten models is first trained on the training split with sensible default settings and scored on the validation split. This untuned pass serves two purposes. It gives a reference point against which the value of tuning can be measured, which becomes Table XXVII in Chapter 6. It also catches configuration errors early, because a model that scores near the baseline at defaults is either badly suited to the data or wired up wrongly, and it is much cheaper to find that out before a tuning run than after.

Each model is then tuned with Optuna against validation macro-F1, using the search spaces and trial budgets set out in Section 5.8. The tuning stage never touches the test set.

The best configuration found for each model is refitted on the training split and scored once on the test set. Predicted labels and predicted probabilities are both stored, because the labels are needed for the confusion matrix and the significance tests while the probabilities are needed for the ROC and precision-recall curves.

Finally the stored predictions are used for the pairwise McNemar tests, and the coefficients of the best linear model are extracted for interpretation.

Two dummy classifiers pass through the identical pipeline. The majority classifier always predicts positive. The stratified classifier predicts at random in proportion to the training class rates. Neither is tuned, since neither has anything to tune, but both are scored by the same evaluation function that scores the real models, which guarantees that the baseline figures are computed the same way as everything they are compared against.

---
## 5.7 The Ten Models

Ten classifiers were implemented, covering six families of machine learning. Each is described below with what it does, why it was included, which feature view it uses, and what it achieved. The results quoted here are the tuned test-set figures reported in full in Chapter 6.

**TABLE XVIII. THE TEN MODELS, THEIR FAMILIES AND FEATURE VIEWS**

| # | Model | Family | Feature view | Why it is in the study |
|---|---|---|---|---|
| 1 | Logistic Regression | Linear | Sparse, 96,755 | Standard linear baseline for sparse text |
| 2 | Linear SVM | Linear | Sparse, 96,755 | Maximum-margin alternative to logistic regression |
| 3 | Multinomial Naive Bayes | Probabilistic | Sparse, 96,755 | Classical text-classification baseline, extremely fast |
| 4 | k-Nearest Neighbours | Instance-based | Dense, 300 | Makes no assumption about boundary shape; deliberate contrast |
| 5 | Random Forest | Bagged trees | Dense, 300 | Standard strong baseline for tabular data |
| 6 | Extra Trees | Bagged trees | Dense, 300 | Tests whether extra randomness helps on projected text |
| 7 | XGBoost | Boosting | Dense, 300 | Most frequently reported winner in comparative studies |
| 8 | LightGBM | Boosting | Dense, 300 | Leaf-wise growth, tests a different boosting strategy |
| 9 | CatBoost | Boosting | Dense, 300 | Symmetric trees and ordered boosting, completes the set |
| 10 | MLP (Neural Network) | Neural | Dense, 300 | Tests whether the class boundary is curved |

### 5.7.1 Logistic Regression

Logistic regression is the simplest classifier in the study and it turned out to be the best. It assigns a weight to every feature, adds up the weighted values for a review, and passes the total through a logistic curve to produce a probability between zero and one.

It suits sparse text for a structural reason rather than an accidental one. With 96,755 features and 13,869 training rows the two classes are almost always separable by a flat boundary, and a flat boundary is exactly what this model draws. It also has a practical advantage that none of the tree models share: its weights can be read directly, so it is possible to ask what the model actually learned. Section 6.8 does exactly that.

Optuna searched the inverse regularisation strength C between 0.05 and 50 on a logarithmic scale, along with whether to weight the classes inversely to their frequency. The best trial used C = 8.034 with `class_weight` set to balanced, and the solver ran to a maximum of 3,000 iterations.

**Result:** accuracy 95.39 percent, macro-F1 0.8951, MCC 0.7914, negative-class recall 0.8567. First on every metric except ROC-AUC, and it fits in 2.15 seconds.

### 5.7.2 Linear Support Vector Machine

The linear SVM also draws a flat boundary but selects it differently. Instead of fitting probabilities it looks for the boundary that leaves the widest empty margin between the two classes. Only the reviews closest to the boundary influence where it sits, so distant points cannot drag it around, which usually makes it steadier than logistic regression when the data is noisy.

A linear kernel was chosen over a radial one because kernel methods scale badly at this feature width and would have made the model impractical to tune. The base implementation returns a signed distance from the boundary rather than a probability, and probabilities are needed for the ROC and precision-recall curves, so the classifier is wrapped in a calibration step using three-fold internal cross-validation to convert its outputs into probabilities.

Optuna searched C between 0.01 and 10 along with the class weighting, and settled on C = 0.233 with `class_weight` set to None.

**Result:** accuracy 95.02 percent, macro-F1 0.8781, MCC 0.7568, and the highest ROC-AUC in the whole study at 0.9762. Its negative-class recall of 0.7556 is a full ten points below logistic regression, and that is what pushes it into second place despite the better ranking score.

### 5.7.3 Multinomial Naive Bayes

Multinomial naive Bayes treats a review as a bag of tokens and asks which class was more likely to have produced that particular bag. It assumes every token is independent of every other one, which is plainly untrue for language, yet the model works well anyway because the false assumption mostly damages its probability estimates rather than the ranking of the two classes.

It was included as the classical baseline for text classification and because it is extremely fast. The only hyperparameter worth tuning is the additive smoothing constant alpha, which decides how much probability mass is reserved for tokens never seen with a given class. Optuna searched it between 0.005 and 3.0 on a logarithmic scale and chose alpha = 0.0273.

**Result:** accuracy 94.25 percent, macro-F1 0.8676, MCC 0.7357, negative-class recall 0.7949, in a fit time of 0.06 seconds. It is the only model in the study whose tuning made it slightly worse than its default settings, losing 0.0047 macro-F1, which is an honest reminder that hyperparameter search on a validation set does not always transfer to a test set.

### 5.7.4 k-Nearest Neighbours

k-nearest neighbours stores the training set and classifies a new review by finding the reviews most similar to it and taking a vote. It has no training phase at all, which is why its fit time in Table XXIX is recorded as 0.00 seconds.

It was included as a deliberate contrast for two reasons. It makes no assumption about the shape of the boundary, so it tests whether the other models are being helped or hindered by their assumptions. And a method that fails on this data tells us something about the data.

It runs on the SVD view, since distances in a 96,755-dimensional sparse space are close to meaningless. Optuna searched the neighbour count from 5 to 75, the vote weighting, and the distance measure, choosing 10 neighbours with uniform weights and cosine distance.

**Result:** accuracy 91.18 percent, macro-F1 0.7474, MCC 0.5163, negative-class recall 0.4382. It is last on every metric, and its 91.18 percent accuracy against an 88.02 percent baseline is the clearest single example in this study of a high accuracy hiding a weak model.

### 5.7.5 Random Forest

A random forest grows many decision trees, each on a random sample of rows and considering a random subset of features at every split, then averages their votes. The randomness is the point. Individual trees overfit badly, but they do not make the same mistakes, so averaging cancels much of the noise.

It was included because it is the standard strong baseline for tabular data and because comparing it against the three boosting models isolates the effect of boosting itself, since all four are tree ensembles on the same 300 features.

Optuna searched tree count, maximum depth, minimum leaf size, feature sampling and class weighting, arriving at 700 trees, maximum depth 40, minimum leaf size 3, `sqrt` feature sampling and `class_weight` set to balanced.

**Result:** accuracy 92.03 percent, macro-F1 0.7725, MCC 0.5669, negative-class recall 0.4775. It finds fewer than half the negative reviews in the test set, and its tuning was the most expensive of all ten models at 757.3 seconds, which is more than a quarter of the entire tuning budget for the ninth-placed result.

### 5.7.6 Extra Trees

Extremely randomised trees follow the same recipe as a random forest but add another layer of randomness. Instead of searching for the best threshold at each split, they draw thresholds at random and keep the best of those draws. Each tree is weaker but faster to grow and less correlated with its neighbours, which sometimes helps the ensemble.

It was included specifically to test whether that extra randomness helps on projected text features, where the input columns are abstract SVD directions rather than interpretable measurements.

Optuna chose 700 trees, maximum depth 40, minimum leaf size 8, 30 percent of features per split and `class_weight` set to balanced_subsample.

**Result:** accuracy 92.60 percent, macro-F1 0.8095, MCC 0.6232, negative-class recall 0.6011. It beats the random forest on every metric, and it showed the single largest improvement from tuning of any model in the study, gaining 0.3327 macro-F1 over its default configuration.

### 5.7.7 XGBoost

XGBoost builds trees one after another, with each new tree trained to correct the mistakes of the trees already built. It adds second-order gradient information and explicit regularisation on the leaf weights, which is why it converges faster and overfits less than plain gradient boosting.

It was included as the first of three boosting libraries and because it is the model most often reported as the winner in comparative studies, including the Olist reference paper [13] where it placed second behind an LSTM. It ran with the histogram tree method for speed.

Optuna searched estimator count, learning rate, depth, row and column subsampling and the positive-class weight, settling on 700 estimators, learning rate 0.118, maximum depth 3, subsample 0.733, column subsample 0.798 and `scale_pos_weight` 0.422.

**Result:** accuracy 94.52 percent, macro-F1 0.8662, MCC 0.7330, negative-class recall 0.7388. Fifth on macro-F1, and the best of the three boosting models.

### 5.7.8 LightGBM

LightGBM is a boosting library built for speed. It grows trees leaf by leaf rather than level by level, always splitting the leaf that promises the largest gain, and it bins continuous features into histograms before training. The result is a model that trains several times faster than XGBoost on the same data at similar accuracy.

It was included both as a second boosting method and to check whether the leaf-wise growth strategy behaves differently on SVD components than the depth-wise one does.

Optuna chose 800 estimators, learning rate 0.109, 31 leaves, a minimum of 49 samples per leaf, column subsample 0.620 and `class_weight` set to balanced.

**Result:** accuracy 94.18 percent, macro-F1 0.8520, MCC 0.7074, negative-class recall 0.6798. Its negative-class recall is noticeably weaker than XGBoost's despite a very similar accuracy, which is a trade-off the balanced class weighting failed to correct.

### 5.7.9 CatBoost

CatBoost is the third boosting library and differs from the other two in how it constructs trees. It builds symmetric trees, meaning every node at the same depth uses the same split condition, which acts as a strong regulariser and makes prediction very fast. It also uses ordered boosting to reduce the subtle target leakage that ordinary boosting introduces when the same rows are used both to compute gradients and to fit the trees that correct them.

It was included to complete the boosting comparison across the three main libraries in current use.

Optuna searched iterations, learning rate, depth, L2 leaf regularisation and automatic class weighting, choosing 500 iterations, learning rate 0.243, depth 4, L2 leaf regularisation 3.156 and `auto_class_weights` set to Balanced.

**Result:** accuracy 93.34 percent, macro-F1 0.8479, MCC 0.6967, negative-class recall 0.7669. Its negative-class recall is the third best in the study, ahead of both the linear SVM and the MLP, but its overall accuracy is the lowest of the three boosting models.

### 5.7.10 Multi-Layer Perceptron

The multi-layer perceptron is a feed-forward neural network. It passes the 300 SVD components through a hidden layer of neurons, applies a non-linear activation to each, and reads a class probability off the output layer.

Unlike the linear models it can represent curved decision boundaries, so it was included to test whether the boundary between positive and negative reviews is in fact curved. Early stopping was enabled with a patience of eight rounds so that training halts once validation performance stops improving.

Optuna searched the hidden layer shape, the L2 penalty, the initial learning rate, the batch size and the iteration cap, choosing a single hidden layer of 128 neurons, alpha 1.06 times ten to the power of minus six, initial learning rate 0.00444, batch size 256 and 120 maximum iterations.

**Result:** accuracy 94.62 percent, macro-F1 0.8698, MCC 0.7399, negative-class recall 0.7528. Third overall, and the best model on the dense feature view, but still behind both linear models on the sparse one. The answer to the question it was included to ask is therefore that the boundary does not appear to be usefully curved, or at least that whatever curvature exists is worth less than the 70 percent of variance the projection threw away.

## 5.8 Hyperparameter Tuning with Optuna

### 5.8.1 Why Optuna

Optuna handles the hyperparameter search for all ten models. It uses a tree-structured Parzen estimator, which builds a probability model of which settings produced good scores and which produced bad ones, then samples the next trial from the region where good scores are more likely. This finds strong settings in far fewer trials than a grid search, which matters when the total budget is forty-five minutes on a laptop.

### 5.8.2 What Is Optimised and Why

Every study optimises macro-F1 on the validation split, never accuracy. The reason is the one given repeatedly in this report: on an 88 to 12 split, optimising accuracy rewards a model for ignoring the minority class. A search told to maximise accuracy would find that switching class weighting off improves accuracy slightly while destroying negative-class recall, and it would take that trade every time. Optimising macro-F1 makes the search protect both classes.

Section 6.5 contains a clean demonstration that this worked. The CatBoost search selected a configuration that scored 0.0030 lower on accuracy than its own default settings while scoring 0.0193 higher on macro-F1. That is not the search failing. It is the search doing precisely what it was told to do.

### 5.8.3 Trial Budgets and Search Spaces

Trial budgets are set per model rather than flat, because the cost of a single fit varies by more than two orders of magnitude across the roster. A naive Bayes fit takes 0.06 seconds while a CatBoost fit takes 26 seconds. An equal budget of, say, 30 trials each would have spent nearly the entire runtime inside CatBoost and the MLP while leaving the cheap models under-searched.

**TABLE XX. OPTUNA SEARCH SPACES AND TRIAL BUDGETS**

| Model | Trials | Parameters searched |
|---|---|---|
| Logistic Regression | 30 | C from 0.05 to 50 log-scale; class_weight |
| Multinomial NB | 30 | alpha from 0.005 to 3.0 log-scale |
| Linear SVM | 25 | C from 0.01 to 10 log-scale; class_weight |
| k-Nearest Neighbours | 15 | n_neighbors 5 to 75; weights; distance metric |
| Random Forest | 15 | n_estimators; max_depth; min_samples_leaf; max_features; class_weight |
| Extra Trees | 15 | n_estimators; max_depth; min_samples_leaf; max_features; class_weight |
| XGBoost | 15 | n_estimators; learning_rate; max_depth; subsample; colsample_bytree; scale_pos_weight |
| LightGBM | 15 | n_estimators; learning_rate; num_leaves; min_child_samples; colsample_bytree; class_weight |
| CatBoost | 12 | iterations; learning_rate; depth; l2_leaf_reg; auto_class_weights |
| MLP (Neural Net) | 12 | hidden_layer_sizes; alpha; learning_rate_init; batch_size; max_iter |
| **Total** | **184** | |

The full search came to 184 trials and 2,728.6 seconds, which is about forty-five minutes on an ordinary laptop CPU with no graphics card.

**TABLE XXI. SELECTED HYPERPARAMETERS FOR EACH MODEL**

| Model | Selected configuration |
|---|---|
| Logistic Regression | C = 8.034, class_weight = balanced, max_iter = 3000 |
| Linear SVM | C = 0.233, class_weight = None, calibrated with 3-fold CV |
| Multinomial NB | alpha = 0.0273 |
| k-Nearest Neighbours | n_neighbors = 10, weights = uniform, metric = cosine |
| Random Forest | 700 trees, max_depth 40, min_samples_leaf 3, max_features sqrt, class_weight balanced |
| Extra Trees | 700 trees, max_depth 40, min_samples_leaf 8, max_features 0.3, class_weight balanced_subsample |
| XGBoost | 700 estimators, lr 0.118, max_depth 3, subsample 0.733, colsample 0.798, scale_pos_weight 0.422 |
| LightGBM | 800 estimators, lr 0.109, 31 leaves, min_child_samples 49, colsample 0.620, class_weight balanced |
| CatBoost | 500 iterations, lr 0.243, depth 4, l2_leaf_reg 3.156, auto_class_weights Balanced |
| MLP (Neural Net) | hidden layer (128), alpha 1.06e-06, lr_init 0.00444, batch 256, max_iter 120 |

One pattern in Table XXI is worth flagging now because Chapter 6 returns to it. Every model that had a class weighting option in its search space had it switched on in the winning trial, with the single exception of the linear SVM. That is nine models out of ten choosing the same setting independently, which is a strong signal about what actually matters on this data.

## 5.9 Evaluation Metrics

Seven metrics are computed for every model. They are described here in the order of importance this project assigns to them, which is deliberately not the order most papers use.

### 5.9.1 Macro-F1 Score

Macro-F1 is the primary metric of this project and every ranking in Chapter 6 is built on it.

The F1 score is the harmonic mean of precision and recall. Macro-averaging means F1 is computed separately for the negative class and the positive class and then averaged with equal weight, regardless of how many reviews are in each. That equal weighting is precisely what this problem needs, because it stops the majority class from drowning out the minority one.

The always-positive baseline makes the value of this metric obvious. It scores an F1 of 0.9363 on the majority class and exactly 0.0000 on the minority class, which averages to 0.4681, even though its accuracy is 88.02 percent. Macro-F1 shows the failure that accuracy conceals. The best model in this study reaches 0.8951, which is 0.4270 above that floor.

### 5.9.2 Matthews Correlation Coefficient

The Matthews correlation coefficient compresses the whole confusion matrix into a single number between minus one and plus one, where plus one is perfect agreement, zero is what random guessing produces, and negative values mean the model is systematically wrong.

All four cells of the confusion matrix enter its formula, so a model can only score highly by getting both classes right at the same time. This makes it very hard to fool. Both baselines confirm the property directly: the majority baseline scores exactly 0.0000 and the stratified baseline scores −0.0005, despite accuracies of 88.02 percent and 79.00 percent. MCC is used throughout this report as the check that a good macro-F1 was not bought with a lopsided trade-off.

### 5.9.3 Recall on the Negative Class

Recall on the negative class answers the question the retailer actually cares about: out of all the genuinely negative reviews in the test set, what fraction did the model catch? It is the number of correctly caught negative reviews divided by 356.

This is the metric with the most direct business meaning, because a missed negative review is a complaint that never reaches anyone. It also separates models that accuracy groups together. Random forest and logistic regression differ by 3.36 accuracy points but by 38 points of negative-class recall, and it is the second gap that decides whether a system is worth deploying.

### 5.9.4 Area Under the ROC Curve

The ROC curve plots the true positive rate against the false positive rate as the decision threshold moves across the probability range, and the area under it measures how well the model ranks reviews rather than how well it labels them at one fixed cut-off. A value of 0.5 is random and 1.0 is perfect.

It is reported here because it is threshold-independent, which makes it a fair way to compare models that may naturally operate at different cut-off points. Both baselines land at 0.5000 and 0.4997 as expected.

### 5.9.5 Area Under the Precision-Recall Curve

PR-AUC summarises the trade-off between precision and recall across all thresholds. It is normally preferred over ROC-AUC when one class is rare, because ROC-AUC counts true negatives and is flattered by a large majority class.

There is an important catch in this study and it is worth stating clearly so the numbers in Table XXVI are not misread. The rare class here is the negative one, while the positive class is the 88.04 percent majority. PR-AUC computed on the positive class therefore has a baseline of 0.8802 rather than something near zero. That is why every model scores above 0.97 on PR-AUC and why those values look uniformly excellent. They are real, but they are not discriminating, and negative-class recall should be read instead.

### 5.9.6 Cohen's Kappa

Cohen's kappa measures how far the model's agreement with the true labels exceeds the agreement that would happen by chance given the class rates. It runs on roughly the same scale as MCC and behaves very similarly on binary problems, which the results confirm: logistic regression scores 0.7903 on kappa against 0.7914 on MCC.

It is reported for completeness and because some readers are more familiar with it than with MCC, but no conclusion in this report rests on it alone.

### 5.9.7 Accuracy

Accuracy is the share of all test reviews classified correctly, and it is reported last on purpose.

On this data the majority baseline already reaches 88.02 percent, so any accuracy figure has to be read as a distance from that number rather than as a value in itself. It is included because readers expect it, because the project target was stated in those terms, and because comparison with earlier work requires it. It is never used to rank models. Where a model's accuracy rises while its negative-class recall falls, this report treats that as a warning rather than an improvement, and Section 6.3 shows two models where exactly that happens.

## 5.10 Control Baselines

Two dummy classifiers are carried through the entire pipeline and scored by the same evaluation code as the ten real models.

The **majority baseline** predicts the positive class for every single review. It learns nothing and reads nothing. Its purpose is to establish the floor that any real result must be measured against, and on this data that floor is high.

The **stratified baseline** predicts at random in proportion to the class rates observed in the training data, so it produces about 88 percent positives and 12 percent negatives at random. It is a slightly more interesting control than the majority classifier because it does produce some negative predictions, and it shows that producing negative predictions at random buys nothing at all.

Both were included from the beginning rather than added afterwards. Adding a baseline after seeing the results invites the temptation to pick the one that flatters them most.

## 5.11 Software Structure of the Project

The project is organised as one notebook supported by a set of Python modules, with every path, seed and column name defined in one configuration file so that nothing can drift out of step.

```
E Commerce/
├── WECR_Research.ipynb        the main notebook, 67 cells
├── WORKFLOW.md                the working plan and measured results
├── _make_notebook.py          regenerates the notebook from source
├── dataset/                   the raw CSV
├── modules/
│   ├── config.py              paths, seeds, column names, all settings
│   ├── data_prep.py           loading, cleaning, target construction
│   ├── eda.py                 all exploratory tables and figures
│   ├── features.py            splitting, TF-IDF, SVD, scaling
│   ├── models.py              the ten model definitions and defaults
│   ├── tuning.py              the Optuna studies and search spaces
│   ├── evaluate.py            metrics, confusion matrices, significance tests
│   ├── exports.py             writing the result workbooks
│   └── workflow_figure.py     draws the workflow diagram
├── figures/                   working copies at 120 dpi
├── paper_figures/             final copies at 300 dpi
└── results/                   seven Excel workbooks
```

The configuration module is the part worth explaining, because it is what makes the results reproducible. The random seed, the split proportions, the TF-IDF settings, the number of SVD components, the tuning metric and every file path are defined there once. No other file contains a hard-coded value for any of them. Changing the seed in one place changes it everywhere, which means a second run either reproduces the first exactly or the difference is traceable to a single line.

Seven Excel workbooks are written to the results folder, one per stage of the pipeline. `01_raw_profile.xlsx` holds the overview, schema, descriptive statistics, cleaning log and neutral-class evidence. `02_eda.xlsx` holds every table behind the exploratory figures. `03_split_and_features.xlsx` holds the split summary, the feature summary and the full 300-row SVD variance table. `04_model_results.xlsx` holds every metric for every model at both the untuned and tuned stages. `05_tuning_history.xlsx` holds all 184 Optuna trials with their parameters and scores. `06_significance_tests.xlsx` holds all 45 McNemar pairs and the Holm-corrected matrix. `PAPER_RESULTS.xlsx` holds the tables in the exact form they appear in this report.

Writing every intermediate result to a workbook was a deliberate choice rather than a convenience. It means every number printed in Chapter 6 can be traced back to the cell that produced it, and it means the tables in this report were copied from generated output rather than typed by hand, which removes a whole class of transcription error.

---

\newpage
# Chapter 6
# Results and Discussion

## 6.1 The Baselines

Every number in this chapter should be read against two controls that were run through the identical evaluation code as the ten real models.

**TABLE XXII. CONTROL BASELINES ON THE TEST SET**

| Baseline | Accuracy | Macro-F1 | MCC | Kappa | ROC-AUC | PR-AUC | Negative recall |
|---|---|---|---|---|---|---|---|
| Majority (always positive) | 0.8802 | 0.4681 | 0.0000 | 0.0000 | 0.5000 | 0.8802 | 0.000 |
| Stratified (random, class-proportional) | 0.7900 | 0.4997 | −0.0005 | −0.0005 | 0.4997 | 0.8802 | 0.118 |

The majority baseline is the important one. It answers "positive" for all 2,972 test reviews. It reads nothing, learns nothing and adjusts to nothing, and it scores 88.02 percent accuracy. It also scores exactly 0.0000 on MCC and 0.000 on negative-class recall, which is the correct description of a model that has never once identified a complaint.

This has a consequence for how the rest of this chapter should be read. An 88.02 percent accuracy is available to a model that learned nothing whatsoever, so the interesting range of this problem is not from zero to a hundred percent. It runs from 88.02 percent to 100 percent. Read that way, the headline result of 95.39 percent closes 61.5 percent of the gap that was actually available, which is a more honest description of the achievement than "95 percent accurate".

The stratified baseline is a useful second control because it does produce negative predictions. It guesses at random in proportion to the training class rates, so roughly twelve percent of its predictions are negative, and it catches roughly twelve percent of the true negatives. Its accuracy is 79.00 percent, which is nine points below the majority baseline, and its MCC is −0.0005, which is zero within rounding. Producing negative predictions at random buys nothing. Only producing the right negative predictions counts, and that is what MCC measures and accuracy does not.

## 6.2 Main Model Comparison

Table XXIII ranks all ten models plus both baselines by macro-F1, which is the primary metric of this study.

**TABLE XXIII. MAIN COMPARISON, TEST SET, RANKED BY MACRO-F1**

| Rank | Model | Family | View | Accuracy | Macro-F1 | MCC | Kappa | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Logistic Regression** | Linear | Sparse | **0.9539** | **0.8951** | **0.7914** | 0.7903 | 0.9755 | 0.9959 |
| 2 | Linear SVM | Linear | Sparse | 0.9502 | 0.8781 | 0.7568 | 0.7562 | **0.9762** | 0.9960 |
| 3 | MLP (Neural Net) | Neural | Dense | 0.9462 | 0.8698 | 0.7399 | 0.7396 | 0.9671 | 0.9940 |
| 4 | Multinomial NB | Probabilistic | Sparse | 0.9425 | 0.8676 | 0.7357 | 0.7352 | 0.9728 | **0.9961** |
| 5 | XGBoost | Boosting | Dense | 0.9452 | 0.8662 | 0.7330 | 0.7324 | 0.9674 | 0.9947 |
| 6 | LightGBM | Boosting | Dense | 0.9418 | 0.8520 | 0.7074 | 0.7042 | 0.9653 | 0.9947 |
| 7 | CatBoost | Boosting | Dense | 0.9334 | 0.8479 | 0.6967 | 0.6959 | 0.9622 | 0.9940 |
| 8 | Extra Trees | Tree | Dense | 0.9260 | 0.8095 | 0.6232 | 0.6194 | 0.9512 | 0.9927 |
| 9 | Random Forest | Tree | Dense | 0.9203 | 0.7725 | 0.5669 | 0.5478 | 0.9489 | 0.9925 |
| 10 | k-Nearest Neighbours | Instance | Dense | 0.9118 | 0.7474 | 0.5163 | 0.4979 | 0.8983 | 0.9771 |
| - | *Baseline (stratified)* | *Control* | - | *0.7900* | *0.4997* | *−0.0005* | *−0.0005* | *0.4997* | *0.8802* |
| - | *Baseline (majority)* | *Control* | - | *0.8802* | *0.4681* | *0.0000* | *0.0000* | *0.5000* | *0.8802* |

Logistic regression finishes first at 0.8951 macro-F1, followed by the linear SVM at 0.8781 and the MLP at 0.8698. All ten models beat both baselines by a wide margin on macro-F1 and MCC, so every one of them learned something real from the text. Two models clear the 95 percent accuracy mark: logistic regression at 95.39 percent and the linear SVM at 95.02 percent, with the MLP next at 94.62 percent.

The most striking feature of this table is what is not at the top. XGBoost, LightGBM and CatBoost are three of the most heavily engineered classifiers available, each the product of years of development by dedicated teams, and all three finish below a logistic regression that trains in 2.15 seconds. The gap is not trivial either. There is 0.0289 macro-F1 between logistic regression and XGBoost, and 0.0472 between logistic regression and CatBoost.

The explanation lies in the feature views, and the table makes it visible if the View column is read alongside the ranking. The three models sitting on the sparse matrix are logistic regression, the linear SVM and naive Bayes, and they occupy positions one, two and four. Every model on the dense SVD view sits below them, with the single exception of the MLP, which squeezes into third.

This is a representation effect rather than a model-quality effect, and Section 5.5.3 already quantified it. The SVD projection keeps 29.61 percent of the variance and discards the rest, and much of what it discards is exactly the rare, highly informative term that decides a review. A word such as "itchy" may appear in forty reviews out of 13,869 and be close to perfectly predictive when it does, but it contributes so little to the total variance of the matrix that no early component will preserve it. A linear model on the full sparse matrix simply gives that word a large weight and uses it. The tree models never see it at all.

It is worth being precise about what this does and does not show. It does not show that gradient boosting is worse than logistic regression at text classification in general. It shows that on this dataset, with these budgets, a linear model given 96,755 features beat boosting models given a 300-dimensional summary of the same information. Section 6.9 says plainly what a fairer comparison would have required.

![Figure 18](paper_figures/fig09_model_comparison.png)

**Figure 18.** Accuracy and macro-F1 by model, with the majority baseline marked. The two panels tell different stories, which is the whole point of showing both.

Figure 18 is worth looking at carefully. In the accuracy panel every model sits in a narrow band between 91 and 96 percent, and the baseline line at 88.02 percent is not far below the worst of them. Read alone, that panel suggests ten models of broadly similar quality. In the macro-F1 panel the same ten models spread from 0.7474 to 0.8951, and the baseline sits far below at 0.4681. The second panel is the honest one, and the difference between the two panels is the central argument of this report compressed into one image.

## 6.3 Per-Class Behaviour and the Accuracy Trap

Table XXIV splits the results by class and shows exactly why accuracy cannot be trusted on this problem.

**TABLE XXIV. PER-CLASS RESULTS ON THE TEST SET (356 NEGATIVE, 2,616 POSITIVE)**

| Model | Neg. precision | Neg. recall | Neg. F1 | Pos. precision | Pos. recall | Pos. F1 |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7801 | **0.8567** | **0.8166** | 0.9802 | 0.9671 | 0.9736 |
| Multinomial NB | 0.7428 | 0.7949 | 0.7680 | 0.9718 | 0.9625 | 0.9671 |
| CatBoost | 0.7036 | 0.7669 | 0.7339 | 0.9679 | 0.9560 | 0.9619 |
| Linear SVM | **0.8152** | 0.7556 | 0.7843 | 0.9671 | 0.9767 | 0.9719 |
| MLP (Neural Net) | 0.7882 | 0.7528 | 0.7701 | 0.9666 | 0.9725 | 0.9695 |
| XGBoost | 0.7898 | 0.7388 | 0.7634 | 0.9648 | 0.9732 | 0.9690 |
| LightGBM | 0.8040 | 0.6798 | 0.7367 | 0.9573 | 0.9774 | 0.9672 |
| Extra Trees | 0.7329 | 0.6011 | 0.6605 | 0.9470 | 0.9702 | 0.9585 |
| Random Forest | 0.7692 | 0.4775 | 0.5893 | 0.9324 | **0.9805** | 0.9558 |
| k-Nearest Neighbours | 0.7156 | 0.4382 | 0.5436 | 0.9274 | 0.9763 | 0.9512 |

**TABLE XXV. CONFUSION MATRIX COUNTS FOR ALL TEN MODELS**

| Model | TN (negatives caught) | FP (negatives missed) | FN | TP | Negatives missed as a share of 356 |
|---|---|---|---|---|---|
| Logistic Regression | **305** | 51 | 86 | 2,530 | 14.3% |
| Multinomial NB | 283 | 73 | 98 | 2,518 | 20.5% |
| CatBoost | 273 | 83 | 115 | 2,501 | 23.3% |
| Linear SVM | 269 | 87 | 61 | 2,555 | 24.4% |
| MLP (Neural Net) | 268 | 88 | 72 | 2,544 | 24.7% |
| XGBoost | 263 | 93 | 70 | 2,546 | 26.1% |
| LightGBM | 242 | 114 | 59 | 2,557 | 32.0% |
| Extra Trees | 214 | 142 | 78 | 2,538 | 39.9% |
| Random Forest | 170 | 186 | 51 | 2,565 | 52.2% |
| k-Nearest Neighbours | 156 | 200 | 62 | 2,554 | 56.2% |

Look at random forest. It scores 92.03 percent accuracy, which sounds respectable and is four points above the baseline, but it recovers only 170 of the 356 negative reviews in the test set. It misses 186 of them, which is more than half.

k-nearest neighbours is worse. It scores 91.18 percent accuracy, finds 156 negatives and misses 200. Both models are past 91 percent accuracy while failing at the one job the system exists to do.

Logistic regression, by contrast, finds 305 of the 356 and misses 51.

Put those two facts side by side. The accuracy difference between logistic regression and random forest is 3.36 percentage points. The difference in negative reviews caught is 135 reviews, which is 37.9 percent of every complaint in the test set. If a retailer chose between these two models on accuracy alone, they would see two numbers three points apart and might reasonably pick either. One of them works and one of them does not.

This is what this report means by the accuracy trap, and it is not a hypothetical concern. It is visible in Table XXIV in two separate places, and it would have been completely invisible in a report that printed only the accuracy column.

A second detail in Table XXIV is worth noting because it shows that the two leading models are not interchangeable. Logistic regression has the better negative-class recall at 0.8567 against 0.7556, but the linear SVM has the better negative-class precision at 0.8152 against 0.7801. In plain terms, logistic regression catches more complaints but raises more false alarms, while the SVM raises fewer false alarms but lets more complaints through.

Which of those is preferable depends entirely on what happens downstream. If flagged reviews are sent to a human reviewer, false alarms are cheap, because the human simply reads a review that turned out to be fine and moves on. In that setting logistic regression is clearly the right choice. If the flag triggers an automatic action, such as pulling a product listing or emailing the supplier, then a false alarm has a real cost and the SVM becomes the more attractive option. This is a deployment decision rather than a modelling one, and it cannot be made from the accuracy column.

![Figure 19](paper_figures/fig10_confusion_matrices.png)

**Figure 19.** Confusion matrices for all ten models. The cell that matters is the one counting negative reviews the model called positive; it ranges from 51 for logistic regression to 200 for k-nearest neighbours.

## 6.4 Ranking Curves and Threshold Behaviour

The metrics in Section 6.2 all evaluate the models at one fixed decision threshold. The curves in this section evaluate them across all thresholds at once, which answers a different and useful question: how well does the model rank reviews, independently of where the cut-off is placed?

**TABLE XXVI. ROC-AUC AND PR-AUC BY MODEL**

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Linear SVM | **0.9762** | 0.9960 |
| Logistic Regression | 0.9755 | 0.9959 |
| Multinomial NB | 0.9728 | **0.9961** |
| XGBoost | 0.9674 | 0.9947 |
| MLP (Neural Net) | 0.9671 | 0.9940 |
| LightGBM | 0.9653 | 0.9947 |
| CatBoost | 0.9622 | 0.9940 |
| Extra Trees | 0.9512 | 0.9927 |
| Random Forest | 0.9489 | 0.9925 |
| k-Nearest Neighbours | 0.8983 | 0.9771 |
| *Baseline (majority)* | *0.5000* | *0.8802* |

Two observations follow from this table.

The first is that the linear SVM ranks marginally better than logistic regression, at 0.9762 against 0.9755. That gap is 0.0007, which on 2,972 test reviews is well inside the range where nothing can be concluded. What it does say is that the SVM is not a worse model in any deep sense. It orders the reviews just as well, and it loses at the fixed threshold rather than in its underlying judgement. A different cut-off, chosen on validation data rather than left at the default, might well close most of the gap between the two.

The second observation is a warning about the PR-AUC column. Every model scores above 0.97, including k-nearest neighbours, which by every other measure in this report is clearly the weakest of the ten. The reason is stated in Section 5.9.5 and it is worth repeating here where the numbers are visible. PR-AUC is computed on the positive class, and the positive class here is the 88.04 percent majority. Its baseline is therefore 0.8802 rather than something near zero, and the whole metric is compressed into the narrow band above that. The column is real but it is not discriminating, and anyone reading this table should ignore it in favour of the ROC column and, better still, the negative-recall column in Table XXIV.

![Figure 20](paper_figures/fig11_roc_pr_curves.png)

**Figure 20.** ROC and precision-recall curves for all ten models. The precision-recall panel is compressed near the top because the positive class is the 88 percent majority.

## 6.5 Effect of Hyperparameter Tuning

Table XXVII compares each model before and after tuning, and the pattern in it is consistent.

**TABLE XXVII. EFFECT OF TUNING, RANKED BY MACRO-F1 GAIN**

| Model | Untuned macro-F1 | Tuned macro-F1 | Gain | Untuned accuracy | Tuned accuracy |
|---|---|---|---|---|---|
| Extra Trees | 0.4768 | 0.8095 | **+0.3327** | 0.8812 | 0.9260 |
| Random Forest | 0.5579 | 0.7725 | +0.2147 | 0.8913 | 0.9203 |
| k-Nearest Neighbours | 0.6790 | 0.7474 | +0.0683 | 0.9058 | 0.9118 |
| XGBoost | 0.8176 | 0.8662 | +0.0487 | 0.9334 | 0.9452 |
| LightGBM | 0.8155 | 0.8520 | +0.0365 | 0.9337 | 0.9418 |
| Logistic Regression | 0.8620 | 0.8951 | +0.0331 | 0.9452 | 0.9539 |
| CatBoost | 0.8286 | 0.8479 | +0.0193 | 0.9364 | 0.9334 |
| MLP (Neural Net) | 0.8578 | 0.8698 | +0.0120 | 0.9428 | 0.9462 |
| Linear SVM | 0.8718 | 0.8781 | +0.0063 | 0.9475 | 0.9502 |
| Multinomial NB | 0.8723 | 0.8676 | −0.0047 | 0.9455 | 0.9425 |

Tuning helped least where the model was already close to correct and most where the default configuration was badly wrong.

The two extreme cases make this clear. Extra trees gained 0.3327 macro-F1, moving from 0.4768 to 0.8095. A macro-F1 of 0.4768 is almost exactly the majority baseline of 0.4681, which means the default configuration had collapsed onto predicting positive for very nearly everything. Class weighting rescued it. Random forest gained 0.2147 for the same reason and from the same starting condition.

At the other end, the linear SVM gained only 0.0063 and multinomial naive Bayes actually lost 0.0047, finishing slightly worse than its own defaults. That last result is worth reporting rather than hiding, because it is a real property of hyperparameter search: a configuration chosen because it scored best on 2,972 validation reviews will sometimes score slightly worse on a different 2,972 test reviews. With a gap that small the honest description is that tuning did nothing measurable for naive Bayes.

CatBoost shows the third pattern. It lost 0.0030 accuracy while gaining 0.0193 macro-F1. That is not a failure of the search. It is the search doing exactly what Section 5.8.2 told it to do, which was to optimise macro-F1 and let accuracy fall where it may. The model traded a small amount of overall correctness for a better balance between the classes, and on this problem that is the right trade.

Logistic regression gained a useful 0.0331 macro-F1 and 0.0650 MCC, and most of that came from a single change: switching `class_weight` to balanced.

**TABLE XXVIII. TUNING COST AND BEST VALIDATION SCORE**

| Model | Trials | Best validation macro-F1 | Tuning time (s) | Seconds per trial |
|---|---|---|---|---|
| Multinomial NB | 30 | 0.8894 | 2.2 | 0.07 |
| k-Nearest Neighbours | 15 | 0.7541 | 15.7 | 1.05 |
| Logistic Regression | 30 | 0.8934 | 39.0 | 1.30 |
| Linear SVM | 25 | 0.8896 | 45.2 | 1.81 |
| LightGBM | 15 | 0.8638 | 197.4 | 13.16 |
| XGBoost | 15 | 0.8636 | 303.9 | 20.26 |
| MLP (Neural Net) | 12 | 0.8771 | 309.2 | 25.77 |
| Extra Trees | 15 | 0.8215 | 403.7 | 26.91 |
| CatBoost | 12 | 0.8703 | 655.0 | 54.58 |
| Random Forest | 15 | 0.7973 | 757.3 | 50.49 |
| **Total** | **184** | | **2,728.6** | |

Table XXVIII shows how uneven the cost of searching is. A naive Bayes trial costs 0.07 seconds. A CatBoost trial costs 54.58 seconds, which is 780 times more. This is the measurement that justified the per-model budgets in Section 5.8.3, and it is worth asking what a flat budget of 30 trials each would have cost: CatBoost alone would have consumed roughly 1,637 seconds and random forest roughly 1,515 seconds, so those two models would have taken more than the entire actual tuning budget between them.

There is an uncomfortable figure in this table that deserves to be pointed out rather than passed over. Random forest consumed 757.3 seconds of search time, more than a quarter of the whole tuning budget, to arrive at a final macro-F1 of 0.7725, which is ninth out of ten. Multinomial naive Bayes consumed 2.2 seconds to arrive at 0.8676, which is fourth. The search cost of a model is not related to how good the model turns out to be.

The single most useful lever across the whole study was class weighting rather than any architectural parameter. Every search space that offered a class weighting option had it selected in the winning trial for the tree and boosting models, and logistic regression chose balanced as well. Only the linear SVM went the other way. This is a practical finding for anyone repeating this kind of work: on an 88 to 12 split, tuning learning rates and tree depths matters far less than telling the model that the two classes are not equally common.

![Figure 21](paper_figures/fig12_tuning_history.png)

**Figure 21.** Optuna search history for all ten models. The rising line is the best score found so far; the scattered points are individual trials. The extra trees and random forest panels show the large early jump described above.

---
## 6.6 Computational Cost

Table XXIX reports how long each tuned model takes to fit once on the 13,869 training rows.

**TABLE XXIX. FIT TIME AND FEATURE VIEW**

| Model | View | Fit time (s) | Macro-F1 | Macro-F1 per second of fitting |
|---|---|---|---|---|
| k-Nearest Neighbours | Dense | 0.00 | 0.7474 | not applicable, no training phase |
| Multinomial NB | Sparse | 0.06 | 0.8676 | 14.46 |
| Linear SVM | Sparse | 1.37 | 0.8781 | 0.64 |
| Logistic Regression | Sparse | 2.15 | 0.8951 | 0.42 |
| MLP (Neural Net) | Dense | 6.64 | 0.8698 | 0.13 |
| LightGBM | Dense | 15.61 | 0.8520 | 0.05 |
| XGBoost | Dense | 16.45 | 0.8662 | 0.05 |
| CatBoost | Dense | 26.09 | 0.8479 | 0.03 |
| Random Forest | Dense | 26.87 | 0.7725 | 0.03 |
| Extra Trees | Dense | 41.12 | 0.8095 | 0.02 |

The cost spread is large and it does not follow the ranking at all.

The best model in the study, logistic regression, fits in 2.15 seconds. The slowest model, extra trees, takes 41.12 seconds, which is nineteen times longer, to produce a macro-F1 that is 0.0856 lower. Naive Bayes fits in 0.06 seconds and finishes fourth, ahead of all three boosting libraries and both tree ensembles.

For a retailer this matters more than it might appear. A model that retrains in seconds can be refreshed every night on the reviews that arrived that day, which keeps it current with new products, new vocabulary and seasonal shifts in what customers complain about. A model that takes half a minute per fit and eleven minutes to re-tune will in practice be retrained rarely, and a model that is retrained rarely drifts. The gap between 2.15 seconds and 41.12 seconds is not important on a single run. It becomes important when it decides whether retraining happens weekly or never.

There is a second cost worth separating from fit time, which is prediction time. None of these models is slow at prediction. Logistic regression scores a review with a single dot product. The boosting models have to evaluate several hundred small trees, which is more work but still measured in microseconds per review. Prediction cost was not a differentiator in this study and is not reported per model for that reason.

![Figure 22](paper_figures/fig16_accuracy_vs_time.png)

**Figure 22.** Macro-F1 against fit time on a logarithmic scale. Green points use the sparse view and blue points use the dense view. The three best models are also among the four cheapest to fit.

## 6.7 Statistical Significance

Ranking models on a single test set is risky, because a gap of a few tenths of a percentage point can easily be noise. On 2,972 test reviews, a difference of 0.37 percentage points is eleven reviews.

We therefore ran McNemar's test on all 45 possible pairs of the ten models, using their predictions on the same 2,972 test reviews, and corrected the resulting p-values with the Holm-Bonferroni procedure to control the family-wise error rate. Of the 45 pairs, 25 differ significantly at the five percent level and 20 do not.

**TABLE XXX. SIGNIFICANTLY DIFFERENT MODEL PAIRS (HOLM-CORRECTED, p < 0.05)**

| Model A | Model B | A only correct | B only correct | p (Holm) |
|---|---|---|---|---|
| Logistic Regression | k-Nearest Neighbours | 193 | 68 | <0.001 |
| Linear SVM | k-Nearest Neighbours | 166 | 52 | <0.001 |
| Logistic Regression | Random Forest | 161 | 61 | <0.001 |
| Linear SVM | Random Forest | 133 | 44 | <0.001 |
| MLP (Neural Net) | k-Nearest Neighbours | 169 | 67 | <0.001 |
| XGBoost | k-Nearest Neighbours | 163 | 64 | <0.001 |
| Logistic Regression | Extra Trees | 128 | 45 | <0.001 |
| LightGBM | k-Nearest Neighbours | 146 | 57 | <0.001 |
| Linear SVM | Extra Trees | 103 | 31 | <0.001 |
| XGBoost | Random Forest | 112 | 38 | <0.001 |
| Multinomial NB | k-Nearest Neighbours | 169 | 78 | <0.001 |
| MLP (Neural Net) | Random Forest | 130 | 53 | <0.001 |
| LightGBM | Random Forest | 95 | 31 | <0.001 |
| XGBoost | Extra Trees | 87 | 30 | <0.001 |
| Logistic Regression | CatBoost | 99 | 38 | <0.001 |
| MLP (Neural Net) | Extra Trees | 113 | 53 | 0.00014 |
| Multinomial NB | Random Forest | 143 | 77 | 0.00034 |
| LightGBM | Extra Trees | 79 | 32 | 0.00035 |
| Linear SVM | CatBoost | 92 | 42 | 0.00062 |
| CatBoost | k-Nearest Neighbours | 165 | 101 | 0.0029 |
| Multinomial NB | Extra Trees | 107 | 58 | 0.0047 |
| Logistic Regression | Multinomial NB | 62 | 28 | 0.0121 |
| XGBoost | CatBoost | 69 | 34 | 0.0186 |
| MLP (Neural Net) | CatBoost | 88 | 50 | 0.0360 |
| Logistic Regression | LightGBM | 83 | 47 | 0.0450 |

**TABLE XXXI. MODEL PAIRS THAT ARE NOT SEPARABLE (HOLM-CORRECTED, p ≥ 0.05)**

| Model A | Model B | A only correct | B only correct | p (Holm) |
|---|---|---|---|---|
| Extra Trees | k-Nearest Neighbours | 130 | 88 | 0.110 |
| CatBoost | Random Forest | 116 | 77 | 0.118 |
| Linear SVM | LightGBM | 62 | 37 | 0.286 |
| Logistic Regression | XGBoost | 69 | 43 | 0.309 |
| Linear SVM | Multinomial NB | 56 | 33 | 0.315 |
| LightGBM | CatBoost | 72 | 47 | 0.417 |
| Multinomial NB | CatBoost | 91 | 64 | 0.515 |
| Logistic Regression | MLP (Neural Net) | 68 | 45 | 0.515 |
| CatBoost | Extra Trees | 83 | 61 | 0.961 |
| Random Forest | k-Nearest Neighbours | 116 | 91 | 1.000 |
| Extra Trees | Random Forest | 59 | 42 | 1.000 |
| Linear SVM | XGBoost | 54 | 39 | 1.000 |
| Logistic Regression | Linear SVM | 36 | 25 | 1.000 |
| Linear SVM | MLP (Neural Net) | 52 | 40 | 1.000 |
| MLP (Neural Net) | LightGBM | 65 | 52 | 1.000 |
| XGBoost | LightGBM | 40 | 30 | 1.000 |
| MLP (Neural Net) | Multinomial NB | 76 | 65 | 1.000 |
| XGBoost | Multinomial NB | 72 | 64 | 1.000 |
| MLP (Neural Net) | XGBoost | 54 | 51 | 1.000 |
| Multinomial NB | LightGBM | 68 | 66 | 1.000 |

The significant differences separate the top group from the bottom. Logistic regression beats k-nearest neighbours, random forest, extra trees and CatBoost with corrected p-values below 0.001, and it beats naive Bayes and LightGBM at the five percent level.

What the test does not support is any claim that logistic regression beats the linear SVM. That pair has a corrected p-value of 1.000, with logistic regression alone correct on 36 reviews and the SVM alone correct on 25. Sixty-one disagreements split 36 to 25 is well within what chance produces. The same holds for logistic regression against the MLP at p = 0.515 and against XGBoost at p = 0.309.

The honest reading of Tables XXX and XXXI is that this study has a top group rather than a single winner. Logistic regression, the linear SVM, the MLP, naive Bayes and XGBoost are not separable at 2,972 test reviews. Logistic regression is reported as the best model in this project because it has the highest macro-F1, the highest MCC and the highest negative-class recall of the ten, and because it fits in 2.15 seconds against 16.45 for XGBoost. But the claim that it is significantly better than XGBoost is not one this data supports, and Section 6.10 lists the size of the negative class as the reason.

There is a related point about the bottom of the table. Random forest and k-nearest neighbours are also not separable from each other, at p = 1.000, and neither is separable from extra trees. So the study has a clear top group and a clear bottom group, with the boundary between them well established, and rather less resolution inside each group than the ranking in Table XXIII might suggest.

![Figure 23](paper_figures/fig13_significance.png)

**Figure 23.** Holm-corrected McNemar p-values for all 45 model pairs. The marked cells are the pairs that genuinely differ.

## 6.8 What the Winning Model Learned

Because logistic regression assigns a readable weight to every one of its 96,755 features, it is possible to ask what it actually keyed on. The answer is more interesting than a list of obviously nice and nasty words.

**TABLE XXXII. TWENTY STRONGEST NEGATIVE COEFFICIENTS**

| Feature | Type | Coefficient |
|---|---|---|
| to love | word bigram | −6.574 |
| wanted to | word bigram | −4.850 |
| not flattering | word bigram | −4.570 |
| poor | word | −4.063 |
| going back | word bigram | −3.929 |
| sad | word | −3.612 |
| unflattering | word | −3.550 |
| cheap | word | −3.398 |
| was | word | −3.364 |
| not | word | −3.333 |
| awful | word | −3.159 |
| not worth | word bigram | −3.124 |
| huge | word | −3.123 |
| you have | word bigram | −3.040 |
| " no" | character n-gram | −2.983 |
| bad | word | −2.975 |
| way | word | −2.970 |
| looks like | word bigram | −2.935 |
| comfortable however | word bigram | −2.920 |
| very disappointed | word bigram | −2.842 |

Some of this list is unsurprising. "Poor", "awful", "cheap", "sad" and "very disappointed" are words a person would guess at without running a model.

The revealing cases are the bigrams whose parts are harmless. The single most negative feature in the entire model is the phrase **"to love"** at a coefficient of −6.574. Read as separate words, "to" is neutral and "love" is the strongest positive word in the model at +4.527. Read as a pair, they signal a specific and very common review structure: a compliment set up only to be withdrawn. The sentences look like "I wanted to love this dress, but the fabric was thin" and "So wanted to love these but they run tiny". The phrase "wanted to" follows at −4.850 for exactly the same reason.

"Going back" at −3.929 is another. Neither word is negative. Together, in a clothing review, they almost always mean the item is being returned.

"Comfortable however" at −2.920 is the same pattern once more. The customer concedes one good property and then reverses. A unigram model sees "comfortable", which is positive, and misses the reversal completely.

**TABLE XXXIII. TWENTY STRONGEST POSITIVE COEFFICIENTS**

| Feature | Type | Coefficient |
|---|---|---|
| but not | word bigram | +5.533 |
| not too | word bigram | +5.467 |
| love | word | +4.527 |
| love this | word bigram | +3.564 |
| "lov" | character n-gram | +3.345 |
| "love" | character n-gram | +3.319 |
| great | word | +3.232 |
| " lov" | character n-gram | +3.114 |
| " love" | character n-gram | +3.098 |
| "ove" | character n-gram | +3.072 |
| nice | word | +2.899 |
| "t! " | character n-gram | +2.888 |
| fun | word | +2.835 |
| cute | word | +2.825 |
| "love " | character n-gram | +2.777 |
| very cute | word bigram | +2.750 |
| size down | word bigram | +2.700 |
| happy | word | +2.659 |
| "ove " | character n-gram | +2.657 |
| like it | word | +2.609 |

The positive side contains a mirror-image surprise, and it is the most interesting single result in this section.

The two strongest positive features in the whole model are **"but not"** at +5.533 and **"not too"** at +5.467. Both contain a negation. A naive word-level system would treat "not" as a negative signal, and indeed the model gives the standalone word "not" a coefficient of −3.333. Yet these two phrases containing it are the model's strongest evidence of a happy customer.

The explanation is in how people write clothing reviews. These phrases almost always appear inside a reassurance: "fitted but not tight", "warm but not bulky", "sheer but not see-through", "not too short", "not too heavy for summer". The customer is heading off a worry the reader might have. The model has learned that a negation immediately following a positive adjective usually qualifies praise rather than reversing it.

This is precisely the pattern that unigram features cannot represent, and it is a concrete demonstration of why the bigram features described in Section 5.5.1 earn their place. It also explains part of the sparse-versus-dense gap in Section 6.2, because these phrase features are exactly the kind of low-variance, high-information signal that a 300-component projection discards.

The character n-grams in Table XXXIII deserve one comment. Six of the twenty strongest positive features are character fragments of the word "love": "lov", "love", " lov", " love", "ove" and "love ". At first glance this looks like waste, six features doing the job of one. It is not. Those fragments also fire on "loved", "loves", "loving", "loooove" and on misspellings that the word-level vocabulary has no entry for. The fragment " lov" catches every one of them. This is character n-grams doing exactly the job Section 2.2.3 said they were included for, and the fact that the model gave them large weights on its own is evidence that they were worth including.

The fragment "t! " at +2.888 is a subtler case. It catches words ending in "t" followed by an exclamation mark, which in review text means "Love it!", "Perfect!", "Great!" and "Bought it!". The exclamation mark is doing real work and no word-level feature would have caught it.

![Figure 24](paper_figures/fig14_top_coefficients.png)

**Figure 24.** The twenty strongest terms for each class in the tuned logistic regression.

## 6.9 Discussion

### 6.9.1 Why a Linear Model Won

The headline finding of this study is that a tuned logistic regression on sparse TF-IDF features beat XGBoost, LightGBM, CatBoost and a neural network. That runs against the common expectation that gradient boosting wins on most problems, and it is worth being clear about why it happened here rather than treating it as a general claim.

Three things combined.

The first is dimensionality. With 96,755 features and 13,869 training rows, the data lives in a space where a flat separating boundary almost certainly exists. In that situation a linear model is the correct assumption for the shape of the problem, not a weaker one. Adding curvature gives the model more ways to fit noise without giving it anything real to fit.

The second is that the boosting models never saw those 96,755 features. They saw a 300-component summary retaining 29.61 percent of the variance, so they were solving a harder version of the same problem. Section 6.2 quantified what the projection discards and Section 6.8 showed what kind of feature it discards: rare, phrase-level, high-information terms of exactly the type that decide a review.

The third is the tuning budget. The linear models received 25 to 30 Optuna trials each while CatBoost received 12, because a CatBoost fit costs roughly 400 times what a naive Bayes fit costs. A fairer comparison would give boosting either the sparse matrix or a much larger trial budget, and this study did neither. We say that plainly rather than claiming a general victory for linear models over boosting.

### 6.9.2 What the Neutral Removal Bought

The second finding concerns the neutral class, and it is the decision that made the 95 percent result reachable at all.

Removing the 2,823 three-star reviews raised the best accuracy from 90.93 percent to 95.56 percent in preliminary runs on the same models. It would be easy to present that as a clever trick, so it is worth stating exactly what the removal does and does not buy.

It does not make the model better at reading English. It removes 12.22 percent of the data whose labels were close to random. A label that is 41.41 percent one way and 58.59 percent the other cannot be learned by any model at all, so removing those rows removes noise rather than signal.

The removal is standard in sentiment research, since both the IMDB and SST-2 benchmarks construct their binary versions the same way, and it is defensible here because we measured the ambiguity rather than assuming it. The honest way to report the result is that this project solves a two-class problem on clearly polarised reviews, not the harder three-class problem, and the accuracy figure should be read with that scope attached.

### 6.9.3 Comparison with Earlier Work

Pandey and colleagues [13] report a best accuracy of 87.09 percent on Brazilian e-commerce satisfaction data, and the figures here are seven to eight points higher. That comparison is not meaningful on its own, because the datasets, the targets and the class ratios are all different, and neither of the reference papers states its majority baseline.

Our 95.39 percent sits 7.37 points above an 88.02 percent baseline. A model scoring 87.09 percent on data with a 60 percent majority class would be a considerably stronger result than ours despite the lower number, because it would have closed a far larger share of the available gap. This is the reason Section 5.9 insists on printing the baseline beside every metric. It is a general point about how to read applied machine learning results rather than a criticism of those two papers specifically.

The transformer results in Section 3.3 are a different kind of comparison. XLM-RoBERTa reaching 0.973 F1 on Indonesian reviews [6] and OnDeBERTa reaching 94.18 percent on Amazon Review Polarity [7] are both on different data, but they indicate roughly where the ceiling sits for this family of task. Our 95.39 percent on a CPU in 2.15 seconds is in the same region, and the gap that remains is exactly what Section 7.2 proposes to test.

### 6.9.4 Practical Deployment Guidance

Table XXXIV summarises which model to choose depending on what the system is for.

**TABLE XXXIV. DEPLOYMENT RECOMMENDATION BY SCENARIO**

| Scenario | Recommended model | Reason |
|---|---|---|
| Catch as many complaints as possible, human reviews the flags | Logistic Regression | Highest negative recall at 0.8567; false alarms are cheap when a person checks them |
| Flag triggers an automatic action | Linear SVM | Highest negative precision at 0.8152; fewer wasted automatic actions |
| Nightly retraining on new reviews | Logistic Regression | Fits in 2.15 seconds; can be refreshed daily without noticeable cost |
| Very tight compute budget | Multinomial NB | Fits in 0.06 seconds for a macro-F1 of 0.8676, fourth place overall |
| Model must be explainable to a non-technical team | Logistic Regression | Coefficients read directly as phrase weights, as in Section 6.8 |
| Threshold will be tuned after deployment | Linear SVM | Best ROC-AUC at 0.9762, so it ranks reviews best across all cut-offs |
| Not recommended in any scenario | k-NN, Random Forest | Both miss over half of all negative reviews despite above-90 percent accuracy |

The last row is the one worth carrying away. Two of the ten models tested would look acceptable on an accuracy-only report and should not be deployed for this task under any circumstances.

## 6.10 Limitations

Five limitations should be kept in mind when reading this report.

**The scope was narrowed by design.** Removing three-star reviews means the system is validated only on reviews that express a clear opinion. A deployed version will meet ambiguous reviews and would need either a third class or a confidence threshold that abstains and routes uncertain cases to a human. Section 7.2 proposes both.

**The dense feature view handicapped seven of the ten models.** The ranking in Table XXIII measures model plus representation rather than model alone. The boosting results in particular should not be read as an upper bound on what those libraries can do on this data, and the first item of future work is to re-run them on the sparse matrix to separate the two effects.

**The tuning budget was uneven.** Linear models received up to 30 trials while CatBoost and the MLP received 12. This was a deliberate response to a real cost constraint, documented in Section 5.8.3, but it means the expensive models were searched less thoroughly than the cheap ones and may have had better configurations available that the search never reached.

**The test set contains only 356 negative reviews.** This is why so many pairwise comparisons in Table XXXI come back non-significant. A larger negative sample would be needed to separate the top five models from each other with any confidence. It also means that per-class metrics on the negative class carry more uncertainty than the four decimal places in the tables imply: a shift of ten reviews moves negative recall by 2.8 points.

**The data comes from one retailer, one country and one product category.** The reviews are written mostly by women reviewing women's clothing sold in the United States. The vocabulary that separates the classes is specific to clothing. Terms such as "itchy", "runs small" and "not flattering" would carry no weight at all on electronics reviews, and a model trained here would need retraining rather than transfer to be used elsewhere. Nothing in this study measures how much of what the model learned is about sentiment and how much is about clothing.

---

\newpage
# Chapter 7
# Conclusion and Future Work

## 7.1 Conclusion

This project compared ten machine learning classifiers on 19,813 women's clothing reviews and found that a tuned logistic regression on sparse TF-IDF features is the strongest of them, reaching 95.39 percent accuracy, macro-F1 0.8951, MCC 0.7914 and recall of 0.8567 on the negative class, against a majority baseline of 88.02 percent accuracy and macro-F1 0.4681. Two models cleared 95 percent accuracy, and all ten beat both control baselines by a wide margin on macro-F1 and MCC, which confirms that the review text carries a strong and learnable signal while the metadata carries almost none.

**TABLE XXXV. OBJECTIVES SET AGAINST OUTCOMES ACHIEVED**

| Objective | Outcome |
|---|---|
| 1. Profile the raw dataset completely and log every cleaning action | Done. Table III profiles the raw file; Table IV logs all four cleaning steps with row counts |
| 2. Test whether three-star reviews belong in a two-class problem | Done. Table V measures a 41.41% recommendation rate at three stars against 98.10% rejection at one star and 99.81% recommendation at five |
| 3. Build a sparse and a dense representation of the same text | Done. 96,755-column TF-IDF and a 300-component SVD projection retaining 29.61% of variance, both fitted on training data only |
| 4. Train and compare ten classifiers under identical conditions | Done. Six families, identical splits, identical evaluation code, results in Table XXIII |
| 5. Tune every model with the same procedure and metric | Done. 184 Optuna trials against validation macro-F1, budgets scaled to per-fit cost, Tables XX and XXVIII |
| 6. Evaluate with metrics that respect the minority class, against controls | Done. Seven metrics per model plus two dummy baselines carried through the same code |
| 7. Test whether differences between models are statistically real | Done. All 45 pairs tested with McNemar and Holm correction; 25 significant, 20 not, Tables XXX and XXXI |

Beyond the numbers, the work makes four points that generalise past this dataset.

**Removing an ambiguous middle class is worth measuring and reporting rather than hiding.** Three-star reviewers here recommended the product only 41.41 percent of the time, and that measurement turns a data-cleaning step into a defensible methodological decision. Any study that drops or keeps a neutral class without testing it is making a choice that affects its headline number and not telling the reader.

**Accuracy without a stated baseline is close to meaningless on imbalanced data.** This is demonstrated twice inside our own results. Random forest reaches 92.03 percent accuracy while missing 186 of 356 negative reviews. k-nearest neighbours reaches 91.18 percent while missing 200. Both look acceptable in an accuracy column and neither should be deployed. Of the fourteen comparable studies summarised in Table A, none printed its own majority baseline.

**A simple linear model on the full sparse feature space can outperform three modern boosting libraries and a neural network at a fraction of the computational cost.** Logistic regression fits in 2.15 seconds and beat CatBoost, which fits in 26.09 seconds and took 655 seconds to tune. The reason is specific and measurable rather than mysterious: the 300-component projection given to the boosting models retains 29.61 percent of the variance and discards exactly the rare phrase features that decide a review, such as "to love" at a coefficient of −6.574 and "but not" at +5.533.

**A study should say when its own ranking is not supported by its data.** McNemar's test with a Holm correction shows that the gap between the first and second placed models in this study has a corrected p-value of 1.000, and that the top five models are not separable at 2,972 test reviews. Logistic regression is reported as the best model because it leads on macro-F1, MCC and negative recall and is the fastest of the top group, but the claim that it is significantly better than XGBoost or the MLP is not one this data supports, and saying so is more useful to a reader than declaring a clean winner.

## 7.2 Future Work

There are six directions worth pursuing, listed roughly in order of how much they would add.

**Fine-tune a transformer.** This is the clearest next step. DistilBERT [33] or DeBERTa-v3 reach 96 to 97 percent on comparable review-sentiment benchmarks, and the transformer studies reviewed in Section 3.3 support that range. Running one here would test whether the errors that remain are genuinely ambiguous reviews or a limit of bag-of-words features. The obstacle is hardware. A single fine-tuning run needs a CUDA graphics card and would take six to ten hours on the machine used for this project, which is why it was placed out of scope rather than attempted.

**Re-run the boosting models on the sparse matrix.** This is the cheapest way to remove the largest confound in the study. Giving XGBoost, LightGBM and CatBoost the full 96,755-column matrix, or a much wider SVD projection, would establish whether their deficit in Table XXIII is a property of the algorithms or an artefact of the representation. The result matters either way. If they close the gap, the conclusion becomes a statement about representations rather than about models. If they do not, the conclusion about linear models on sparse text becomes considerably stronger.

**Bring the three-star reviews back.** The 2,823 reviews removed in Section 5.3 will exist in any deployed setting, so a production system has to do something with them. Two options are worth testing. The first is a genuine three-class model, which would be harder and would score lower but would match the real problem. The second is an abstain option, where the classifier reports a confidence and any review below a threshold is routed to a human instead of being labelled. The second option is probably more useful in practice, because it lets the system be accurate on what it is confident about and honest about the rest.

**Tune the decision threshold rather than leaving it at 0.5.** Section 6.4 showed that the linear SVM ranks reviews marginally better than logistic regression while scoring lower at the fixed cut-off. That suggests some of the gap between the two is threshold placement rather than model quality. Choosing the threshold on validation data, to maximise macro-F1 or to hit a target negative-class recall, is nearly free and might change the ranking.

**Test on a different retailer and a different product category.** Section 6.10 lists domain specificity as a limitation and there is no way to measure it from inside this dataset. Applying the trained model to reviews from another shop, and then to reviews of a completely different product type such as electronics, would show how much of what it learned is about sentiment and how much is about clothing. Our expectation, based on the coefficient tables in Section 6.8, is that a good deal of it is about clothing.

**Move from polarity to aspect-level sentiment.** The aspect-based approach of Quoc-Hung and colleagues [4] produces a more useful output than a single label, because a review can be positive about the fabric and negative about the sizing at the same time, and a retailer needs to know which. This is a larger change than the others, since it requires aspect annotations that this dataset does not have, but it is the direction that would add the most value to the retailer the system is built for.

---

\newpage

# Chapter 8
# Bibliography

[1] F. A. M. Mahmud, S. B. R. A. Mullick and T. C. M. Anas, "Sentiment analysis of women's clothing reviews on e-commerce platforms: a machine learning approach," *IEEE Access*, 2023.

[2] I. Yabes, N. Aryanti, R. J. Pradana, K. E. Setiawan and M. F. Hasani, "Classifying and predicting the rating sentiment of women's e-commerce clothing reviews: a comparative study using SVM, ANN, and BERT models," in *Proc. 5th Int. Conf. Cybernetics and Intelligent System (ICORIS)*, IEEE, 2023, pp. 1-6.

[3] N. U. Prince, M. N. H. Shawon, Z. K. Shahed, R. I. Nupur, F. A. Mele and M. A. Al Mamun, "E-commerce clothing review analysis by advanced ML algorithms," *World Journal of Advanced Research and Reviews*, vol. 24, no. 1, pp. 1680-1690, 2024.

[4] P. Quoc-Hung, D. Van-Dan, L. Huu-Loi, L. Thi-Viet-Huong, N. Thu-Ha, P. Xuan-Hieu, N. Minh-Tien and P. Ngoc-Hung, "Aspect-based sentiment analysis of clothing reviews in Vietnamese e-commerce," in *Proc. 38th Pacific Asia Conf. Language, Information and Computation*, 2024, pp. 231-238.

[5] V. R. Hananto, S. Kim, M. Kovacs, U. Serdült and V. Kryssanov, "A machine learning approach to analyze fashion styles from large collections of online customer reviews," in *Proc. 6th Int. Conf. Business and Industrial Research (ICBIR)*, IEEE, 2021, pp. 153-158.

[6] A. A. Putra and K. M. Lhaksmana, "Comparison of IndoBERT and XLM-RoBERTa performance in e-commerce product review sentiment classification," in *Proc. Int. Conf. Advancement in Data Science, E-learning and Information System (ICADEIS)*, IEEE, 2026, pp. 1-6.

[7] J. Liu, W. Gao, M. Ou, B. Fu, X. Huang and Y. Huan, "OnDeBERTa: an ordered neuron memory enhanced DeBERTa model for e-commerce review sentiment classification," in *Proc. Int. Conf. Computer Intelligence and Software Engineering (CICSE)*, IEEE, 2026, pp. 400-405.

[8] R. S. Kiziltepe, E. Ezin, Ö. Yentür, A. M. Basbrain and M. Karakus, "Advancing sentiment analysis for low-resource languages using fine-tuned LLMs: a case study of customer reviews in Turkish language," *IEEE Access*, vol. 13, pp. 77382-77394, 2025.

[9] C. Khan, A. Adnan and S. Anwar, "Empowering trust in digital platforms: sentiment analysis and emotion AI for fake reviews detection," *IEEE Transactions on Computational Social Systems*, 2025.

[10] B. Kumar, V. S. Badiger and A. D. Jacintha, "Sentiment analysis for products review based on NLP using lexicon-based approach and RoBERTa," in *Proc. Int. Conf. Intelligent and Innovative Technologies in Computing, Electrical and Electronics (IITCEE)*, IEEE, 2024, pp. 1-6.

[11] V. Adarsh, V. Mohla and R. K. Mahto, "Sentiment classification of product reviews using machine learning," in *Proc. Int. Conf. Recent Advances in Electrical, Electronics, Ubiquitous Communication, and Computational Intelligence (RAEEUCCI)*, IEEE, 2024, pp. 1-6.

[12] L. G. Atlas, D. Arockiam, A. Muthusamy, B. Balusamy, S. Selvarajan, T. Al-Shehari and N. A. Alsadhan, "A modernized approach to sentiment analysis of product reviews using BiGRU and RNN based LSTM deep learning models," *Scientific Reports*, vol. 15, no. 1, p. 16642, 2025.

[13] T. N. Pandey, A. Vasudev, D. Sagayanathan, G. Anjan, D. Arshad and S. S. Patra, "Predicting customer satisfaction in Brazil e-commerce: a comparative study of machine learning techniques," in *Proc. Int. Conf. Computing, Communication, and Intelligent Systems (ICCCIS)*, 2023.

[14] M. Debbagh, Y. Madani and B. Bouikhalene, "Leveraging multiple models to enhance sales demand forecasting in Brazilian e-commerce," University Sultan Moulay Slimane, Beni Mellal, Morocco.

[15] P. A. Sunarya, U. Rahardja, S. C. Chen, Y. M. Lic and M. Hardini, "Deciphering digital social dynamics: a comparative study of logistic regression and random forest in predicting e-commerce customer behavior," *Journal of Applied Data Sciences*, vol. 5, no. 1, pp. 100-113, 2024.

[16] N. Kalcheva, G. Marinova and M. Todorova, "Comparative analysis of the Bernoulli and multinomial naive Bayes classifiers for text classification in machine learning," in *Proc. Int. Conf. Automatics and Informatics (ICAI)*, IEEE, 2023, pp. 28-31.

[17] B. Pang, L. Lee and S. Vaithyanathan, "Thumbs up? Sentiment classification using machine learning techniques," in *Proc. Conf. Empirical Methods in Natural Language Processing (EMNLP)*, 2002, pp. 79-86.

[18] A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng and C. Potts, "Learning word vectors for sentiment analysis," in *Proc. 49th Annual Meeting of the Association for Computational Linguistics*, 2011, pp. 142-150.

[19] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Ng and C. Potts, "Recursive deep models for semantic compositionality over a sentiment treebank," in *Proc. EMNLP*, 2013, pp. 1631-1642.

[20] G. Salton and C. Buckley, "Term-weighting approaches in automatic text retrieval," *Information Processing and Management*, vol. 24, no. 5, pp. 513-523, 1988.

[21] N. Halko, P. G. Martinsson and J. A. Tropp, "Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions," *SIAM Review*, vol. 53, no. 2, pp. 217-288, 2011.

[22] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001.

[23] P. Geurts, D. Ernst and L. Wehenkel, "Extremely randomized trees," *Machine Learning*, vol. 63, no. 1, pp. 3-42, 2006.

[24] T. Chen and C. Guestrin, "XGBoost: a scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785-794.

[25] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye and T.-Y. Liu, "LightGBM: a highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems 30*, 2017, pp. 3146-3154.

[26] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush and A. Gulin, "CatBoost: unbiased boosting with categorical features," in *Advances in Neural Information Processing Systems 31*, 2018, pp. 6638-6648.

[27] T. Akiba, S. Sano, T. Yanase, T. Ohta and M. Koyama, "Optuna: a next-generation hyperparameter optimization framework," in *Proc. 25th ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2019, pp. 2623-2631.

[28] D. Chicco and G. Jurman, "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation," *BMC Genomics*, vol. 21, no. 6, 2020.

[29] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153-157, 1947.

[30] S. Holm, "A simple sequentially rejective multiple test procedure," *Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65-70, 1979.

[31] T. G. Dietterich, "Approximate statistical tests for comparing supervised classification learning algorithms," *Neural Computation*, vol. 10, no. 7, pp. 1895-1923, 1998.

[32] F. Pedregosa et al., "Scikit-learn: machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.

[33] V. Sanh, L. Debut, J. Chaumond and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," in *Proc. NeurIPS EMC^2 Workshop*, 2019.

[34] nicapotato, "Women's E-Commerce Clothing Reviews," Kaggle, 2018. [Online]. Available: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

---

\newpage

# Chapter 9
# Individual Contribution

**Prerona Karmakar — 23051689**

Prerona Karmakar led the data stage and the sparse-feature side of the study. She wrote the loading and cleaning pipeline, including the four-step cleaning sequence recorded in Table IV, and produced the raw data profile in Table III. She carried out the analysis that justified removing the neutral class, cross-tabulating the recommendation flag against every rating level to produce the 41.41 percent figure in Table V that the whole design of the study rests on.

She built the complete exploratory analysis reported in Section 5.2, including the rating distribution, the class balance, the review-length and age comparisons by class, the positive-rate breakdowns across division, department and product class, and the correlation matrix that established that the metadata carries no usable signal. She produced the word clouds and the log-odds discriminative-term tables.

On the modelling side she implemented the sparse feature pipeline, comprising the word and character TF-IDF vectorisers, their union into the 96,755-column matrix, and the training-only fitting discipline that keeps the evaluation clean. She implemented and tuned the three models that run on that view: logistic regression, the linear support vector machine with its calibration wrapper, and multinomial naive Bayes. She extracted and interpreted the coefficients of the winning model, which became Section 6.8 and Tables XXXII and XXXIII.

For the report she wrote Chapter 1, Chapter 3, Section 5.1 through Section 5.5, and Section 6.8, and prepared the front matter.

**Aadi Poddar — 23051641**

Aadi Poddar led the dense-feature side, the tuning framework and the evaluation stage. He implemented the stratified three-way split with its fixed seed, the truncated SVD projection and the standardisation step, and produced the explained-variance analysis in Table XVII that quantified what the projection costs at 29.61 percent of the original variance.

He implemented and tuned the seven models that run on the dense view: k-nearest neighbours, random forest, extra trees, XGBoost, LightGBM, CatBoost and the multi-layer perceptron. He designed the Optuna tuning framework used by all ten models, including the per-model trial budgets in Table XX that were scaled to per-fit cost, and the decision to optimise validation macro-F1 rather than accuracy throughout.

He built the evaluation module that computes all seven metrics, both dummy baselines and the per-class breakdowns, and wrote the statistical testing stage covering all 45 McNemar pairs with the Holm-Bonferroni correction reported in Tables XXX and XXXI. He produced the model comparison charts, the confusion matrix grid, the ROC and precision-recall curves, the tuning history panels and the significance heatmap, and wrote the seven Excel export workbooks that every table in this report was generated from.

For the report he wrote Chapter 2, Chapter 4, Section 5.6 through Section 5.11, Section 6.1 through Section 6.7, Section 6.9 and Section 6.10, and Chapter 7.

**Joint work**

Both members jointly designed the study, agreed the metric hierarchy before any model was run, reviewed each other's code and results, and re-ran the full pipeline whenever a design decision changed so that the effect of the change could be measured rather than argued about. The discussion, conclusion and future-work sections were written together.

---
