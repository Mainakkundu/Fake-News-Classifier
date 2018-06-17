#################################
# FAKE NEWS CLASSIFIER PROJECT
#################################

# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Print the head of df
print(df.head())

# Create a series to store the labels: y
y = df.label

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words='english')

# Transform the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test data using only the 'text' column values: count_test 
count_test = count_vectorizer.transform(X_test)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])

#--------- TfIDf VECTOR ----------------
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])

'''

<script.py> output:
    ['000', '000ft', '000km', '003', '01', '02', '027', '033', '04', '05']
    [[0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.03834063 0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]
     [0.         0.         0.         ... 0.         0.         0.        ]]

In [1]: 
'''
#--Inspecting the vectors---
'''
To get a better idea of how the vectors work, you'll investigate them by converting them into pandas DataFrames.

Here, you'll use the same data structures you created in the previous two exercises (count_train, count_vectorizer, tfidf_train, tfidf_vectorizer) as well as pandas, which is imported as pd.
'''
# Create the CountVectorizer DataFrame: count_df
count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

# Print the head of count_df
print(count_df.head())

# Print the head of tfidf_df
print(tfidf_df.head())

# Calculate the difference in columns: difference
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)

# Check whether the DataFrames are equal
print(count_df.equals(tfidf_df))
'''
<script.py> output:
       000  10  100  11  114th  11th  12  120  13  133    ...     œexplosiveâ  \
    0    0   0    0   0      0     0   0    0   0    0    ...               0   
    1    0   0    0   0      0     0   0    0   0    0    ...               0   
    2    0   1    0   0      0     0   0    0   1    0    ...               0   
    3    0   1    0   0      0     0   0    0   0    0    ...               0   
    4    0   0    0   0      0     0   0    0   0    0    ...               0   
'''


#--------------------------------------------------
# TRAIN THE MODEL with COUNTVECTORIZER DATAFRAME 
#-------------------------------------------------
# Import the necessary modules
from sklearn import metrics 
from sklearn.naive_bayes import MultinomialNB

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(count_train,y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test,pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test,pred,labels=['FAKE','REAL'])
print(cm)
'''
<script.py> output:
    0.893352462936394
    [[ 865  143]
     [  80 1003]]
'''

#-------------------------------------------
#TRAIN WITH TFIDFVECTORIZER 
#-------------------------------------------
# Create a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()

# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix: cm
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)
'''
0.8565279770444764
[[ 739  269]
 [  31 1052]]
 '''
#--------------------------------------------------
#MULTINOMINAL WITH VARIOUS ALPHAS 
#--------------------------------------------------

# Create the list of alphas: alphas
alphas = np.arange(0,1,0.1)

# Define train_and_predict()
def train_and_predict(alpha=alphas):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB()
    # Fit to the training data
    nb_classifier.fit(tfidf_train,y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    # Compute accuracy: score
    score = metrics.accuracy_score(pred,y_test)
    return score

# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()
	
'''
Score:  0.8857006217120995
    
    Alpha:  0.5
    Score:  0.8842659014825442
    
    Alpha:  0.6000000000000001
    Score:  0.874701099952176
    
    Alpha:  0.7000000000000001
    Score:  0.8703969392635102
    
    Alpha:  0.8
    Score:  0.8660927785748446
    
    Alpha:  0.9
    Score:  0.8589191774270684

'''
#---------- Feature Extraction -------------
# Get the class labels: class_labels
class_labels = nb_classifier.classes_

# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])
'''
FAKE [(-12.641778440826338, '0000'), (-12.641778440826338, '000035'), (-12.641778440826338, '0001'), (-12.641778440826338, '0001pt'), (-12.641778440826338, '000km'), (-12.641778440826338, '0011'), (-12.641778440826338, '006s'), (-12.641778440826338, '007'), (-12.641778440826338, '007s'), (-12.641778440826338, '008s'), (-12.641778440826338, '0099'), (-12.641778440826338, '00am'), (-12.641778440826338, '00p'), (-12.641778440826338, '00pm'), (-12.641778440826338, '014'), (-12.641778440826338, '015'), (-12.641778440826338, '018'), (-12.641778440826338, '01am'), (-12.641778440826338, '020'), (-12.641778440826338, '023')]
REAL [(-6.790929954967984, 'states'), (-6.765360557845786, 'rubio'), (-6.751044290367751, 'voters'), (-6.701050756752027, 'house'), (-6.695547793099875, 'republicans'), (-6.6701912490429685, 'bush'), (-6.661945235816139, 'percent'), (-6.589623788689862, 'people'), (-6.559670340096453, 'new'), (-6.489892292073901, 'party'), (-6.452319082422527, 'cruz'), (-6.452076515575875, 'state'), (-6.397696648238072, 'republican'), (-6.376343060363355, 'campaign'), (-6.324397735392007, 'president'), (-6.2546017970213645, 'sanders'), (-6.144621899738043, 'obama'), (-5.756817248152807, 'clinton'), (-5.596085785733112, 'said'), (-5.357523914504495, 'trump')]
'''
#------------------------------------------------
