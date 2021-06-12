#!/usr/bin/env python
# coding: utf-8

# # Mini-project
Group 10 (individual): Nguyễn Phan Khánh Linh MAMAIU19010
Email: nguyenphankhanhlinh2001@gmail.com
# https://www.analyticsvidhya.com/blog/2018/10/stepwise-guide-topic-modeling-latent-semantic-analysis/
# https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python
# https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/ml-decision-tree/tutorial/
# https://www.kaggle.com/carrie1/decision-tree-classification-using-zoo-animals?fbclid=IwAR2YlizpMr6vNFdhJeQNZJR2PiPjUEEzlT3DLoCFbkcEnroMRuNcL36CAKM

# # Problem 1.5: Decision Trees (elective)

# Uses the Decision Trees method and the learned lessons to mine the animal Zoo dataset available at http://archive.ics.uci.edu/ml/datasets/Zoo. The solution must be evaluated by the test dataset and real self-generated dataset.

# # 1. Data pre-processing: 

# In[1]:


import pandas as pd 
df = pd.read_csv('/Users/nguyenphankhanhlinh/Downloads/zoo-4.csv')
df.info() #view the information of the dataframe
print ('-------------------')
df.head() #list first 5 lines


# In[2]:


df2 = pd.read_csv('/Users/nguyenphankhanhlinh/Downloads/class.csv')
df2.info()
print ('----------------')
df2.head()


# In[3]:


df3 = df.merge(df2,how='left',left_on='class_type',right_on='Class_Number') #Using merge function to merging and concatenating of data
df3.head(5) 


# In[4]:


g = df3.groupby(by='Class_Type')['animal_name'].count() #count how many animals there are in this category
g 


# # 2. Data post processing:

# # A. Sort bars

# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(df3['Class_Type'],label="Count",
             order = df3['Class_Type'].value_counts().index) #sort bars
plt.show()


# # B. Facetgrid
Using the FacetGrid from Seaborn, we can look at the columns to help us understand what features may be more helpful than others in classification
# In[6]:


feature_names = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed',
                 'backbone','breathes','venomous','fins','legs','tail','domestic']

df3['ct'] = 1

for f in feature_names:
    g = sns.FacetGrid(df3, col="Class_Type",  row=f, hue="Class_Type")
    g.map(plt.hist, "ct")
    g.set(xticklabels=[])
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f)


# # C. Heatmap

# In[7]:


gr = df3.groupby(by='Class_Type').mean()
columns = ['class_type','Class_Number','Number_Of_Animal_Species_In_Class','ct','legs'] #will handle legs separately since it's not binary 
gr.drop(columns, inplace=True, axis=1)
plt.subplots(figsize=(10,4))
sns.heatmap(gr, cmap="YlGnBu")


# # D. Stripplot

# In[8]:


sns.stripplot(x=df3["Class_Type"],y=df3['legs']) #show all observations along with some representation of the underlying distribution.


# # 3.	Data processing with algorithm and implementation 

# # A. Decision tree works if we use all of the features available to us and training with 20% of the data

# In[9]:


#specify the inputs (x = predictors, y = class)
X = df[feature_names]
y = df['class_type'] #there are multiple classes in this column

#split the dataframe into train and test groups
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.2, test_size=.8)

#specify the model to train with
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning) #ignores warning that tells us dividing by zero equals zero

#let's see how well it worked
pred = clf.predict(X_test)
print('Accuracy of classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))


# In[10]:


df3[['Class_Type','class_type']].drop_duplicates().sort_values(by='class_type') #this is the order of the labels in the confusion matrix above


# # Features were the most important in this model

# In[11]:


imp = pd.DataFrame(clf.feature_importances_)
ft = pd.DataFrame(feature_names)
ft_imp = pd.concat([ft,imp],axis=1)
ft_imp.columns = ['Feature', 'Importance']
ft_imp.sort_values(by='Importance',ascending=False)


# # B.  Decision tree works if we reduced the training set size to 10%

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.1, test_size=.9) 

clf2 = DecisionTreeClassifier().fit(X_train, y_train)
pred = clf2.predict(X_test)
print('Accuracy of classifier on test set: {:.2f}'
     .format(clf2.score(X_test, y_test)))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))


# # Features were the most important in this model

# In[13]:


imp2 = pd.DataFrame(clf2.feature_importances_)
ft_imp2 = pd.concat([ft,imp2],axis=1)
ft_imp2.columns = ['Feature', 'Importance']
ft_imp2.sort_values(by='Importance',ascending=False)


# # C. Let's go back to 20% in the training group and focus on visible features of the animals

# In[14]:


visible_feature_names = ['hair','feathers','toothed','fins','legs','tail']

X = df[visible_feature_names]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.2, test_size=.8)

clf3= DecisionTreeClassifier().fit(X_train, y_train)

pred = clf3.predict(X_test)
print('Accuracy of classifier on test set: {:.2f}'
     .format(clf3.score(X_test, y_test)))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))


# # Features were the most important in this model

# In[15]:


imp3= pd.DataFrame(clf3.feature_importances_)
ft = pd.DataFrame(visible_feature_names)
ft_imp3 = pd.concat([ft,imp3],axis=1)
ft_imp3.columns = ['Feature', 'Importance']
ft_imp3.sort_values(by='Importance',ascending=False)


# # D.  Using the same train/test groups and visible features group as above.

# If the dataset were larger, reducing the depth size of the tree would be useful to minimize memory required to perform the analysis. Below I've limited it to two still using the same train/test groups and visible features group as above.

# In[16]:


clf4= DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)

pred = clf4.predict(X_test)
print('Accuracy of classifier on test set: {:.2f}'
     .format(clf4.score(X_test, y_test)))
print()
print(confusion_matrix(y_test, pred))
print()
print(classification_report(y_test, pred))


# # Features were the most important in this model

# In[17]:


imp4= pd.DataFrame(clf4.feature_importances_)
ft_imp4 = pd.concat([ft,imp3],axis=1)
ft_imp4.columns = ['Feature', 'Importance']
ft_imp4.sort_values(by='Importance',ascending=False)


# In[18]:


columns = ['Model','Test %', 'Accuracy','Precision','Recall','F1','Train N']
df_ = pd.DataFrame(columns=columns)

df_.loc[len(df_)] = ["Model 1",20,.78,.80,.78,.77,81] 
df_.loc[len(df_)] = ["Model 2",10,.68,.62,.68,.64,91]
df_.loc[len(df_)] = ["Model 3",20,.91,.93,.91,.91,81]
df_.loc[len(df_)] = ["Model 4",20,.57,.63,.57,.58,81]
ax=df_[['Accuracy','Precision','Recall','F1']].plot(kind='bar',cmap="YlGnBu", figsize=(10,6))
ax.set_xticklabels(df_.Model)


# # Problem 1.6 : Latent Semantic Analysis (mandatory)

# Each group uses the Latent Semantic Analysis method and the learned lessons to seek similar documents from the text dataset to the provided queries. The field-specific dataset is provided by the Instructor. The solution must include text data pre-processing, term-document matrix generation, and singular value decomposition application.

# # 1. Import the libraries and the dataset

# # A. Import the libraries

# In[19]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option("display.max_colwidth", 200)


# # B. Import the dataset

# In[20]:


dataset=pd.read_csv('/Users/nguyenphankhanhlinh/Downloads/datafiles/group10.csv')
dataset.head(3)


# In[21]:


dataset.columns = ['1','2','3','4','Detail','6','7','8','9','10','11','12','13','14','15','16','Description']
dataset.head(3)


# In[22]:


dataset=dataset[['Detail','Description']]
dataset.head()


# In[23]:


dataset.shape


# # 2. Data Preprocessing

# In[24]:


preprocessing_dataset= dataset
preprocessing_dataset.iloc[:5]


# # A. Lower-casting ( Make all text lowercase ) 

# In[25]:


preprocessing_dataset['Clean_First'] = preprocessing_dataset['Description'].apply(lambda x: x.lower())
preprocessing_dataset[['Description','Clean_First']].iloc[:5]


# # B.  Removing everything except alphabets ( Removal of punctuation )

# In[26]:


preprocessing_dataset['Clean_Second'] = preprocessing_dataset['Clean_First'].str.replace("[^a-zA-Z#]", " ")
preprocessing_dataset[['Clean_First','Clean_Second']].iloc[:5]


# # C.  Removing short words ( Remove words with length less than or equal to 3 )

# In[27]:


preprocessing_dataset['Clean_Third'] = preprocessing_dataset['Clean_Second'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
preprocessing_dataset[['Clean_Second','Clean_Third']].iloc[:5]


# # D. Remove stop-words

# In[28]:


# We must import NLTK to remove the stopword
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
", ".join(stopwords.words('english'))
STOPW = set(stopwords.words('english'))
def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPW])


# In[29]:


preprocessing_dataset["Clean_Fourth"] = preprocessing_dataset["Clean_Third"].apply(lambda text: remove_stopwords(text))
preprocessing_dataset[['Clean_Third','Clean_Fourth']].iloc[:5]


# # E. Steming

# In[30]:


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


# In[31]:


preprocessing_dataset["Clean_Fifth"]=preprocessing_dataset["Clean_Fourth"].apply(lambda text: stem_words(text))
preprocessing_dataset[['Clean_Fourth','Clean_Fifth']].iloc[:5]


# # F. Comparison between initial description and clean_description

# In[32]:


preprocessing_dataset[['Description','Clean_Fifth']].iloc[:5]


# # 3. Document Term matrix

# In[33]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[34]:


vectorizer = TfidfVectorizer(stop_words='english', max_features= 2000, max_df = 0.5, smooth_idf=True)


# In[35]:


doc_term_matrix= vectorizer.fit_transform(preprocessing_dataset['Clean_Fifth'])


# In[36]:


doc_term_matrix.shape


# # 4. FIND OPTIMAL TOPICS USING EXPLAINED VARIANCE RATIO

# In[37]:


from sklearn.decomposition import TruncatedSVD


# In[38]:


# Program to find the optimal number of components for Truncated SVD
n_comp = [5,15,30, 60, 90, 180, 250,350,500,800,1000] # list containing different values of components
explained = [] # explained variance ratio for each component of Truncated SVD
for x in n_comp:
    svd = TruncatedSVD(n_components=x, algorithm='randomized', n_iter=100)
    svd.fit(doc_term_matrix)
    explained.append(svd.explained_variance_ratio_.sum())
    print("Number of components = %r and explained variance = %r"%(x,svd.explained_variance_ratio_.sum()))


# In[39]:


plt.plot(n_comp, explained)
plt.xlabel('Number of components')
plt.ylabel("Explained Variance")
plt.title("Plot of Number of components v/s explained variance")
plt.show()


# # 5. SINGULAR-VALUE-DECOMPOSITION (SVD)

# In[40]:


# SVD represent documents and terms in vectors 
svd_model = TruncatedSVD(n_components=800, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(doc_term_matrix)


# In[41]:


# print out some some topics of SVD model

terms = vectorizer.get_feature_names()

for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    terms=[]
    for t in sorted_terms:
        terms.append(t[0])
    print("Topic "+str(i)+": ",terms)


# In[ ]:





# In[ ]:




