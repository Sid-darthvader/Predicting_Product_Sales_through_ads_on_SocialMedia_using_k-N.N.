# K-Nearest Neighbors (K-N.N.)
#Predicting product sales through ads delivered on Social Networking Sites using K.N.N.
'''
In simpler words we tell whether a user on Social Networking site after clicking the ad's displayed on the website,
end's up buying the product or not. This could be really helpful for the company selling the product.
Lets say that its a car company which has paid the social networking site(For simplicity we'll assume its Facebook from now on)
to display ads of its newly launched car.Now since the company relies heavily on the success of its newly launched car it would leave no stone unturned while trying to advertise the car.
Well then whats better than advertising it on the most popular platform right now.But what if we only advertise it to the correct crowd.
This could help in boosting sales as we will be showing the ad of the car only to selected crowd.
So this is where you come in...
The Car company has hired you as a Data Scientist to find out the correct crowd, to which you need to advertise the car
and find out the people who are most likely to buy the car based on certain features which describe the type of users who have bought the car
previously by clicking on the ad.

'''
'''
A Gentle Introduction...

'''
"""
HOW DOES THE ALGORITHM WORK?
Step.1)Choose the number K of neighbours
Step.2)Take the K nearest neighbours of the new data point, according to Euclidean Distance
Step.3)Among these K neighbours, count the number of data points in each category
Step.4)Assign the new data point to the category where you counted the most neighbours
......Now your Model is Ready!!

"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
"""
The Dataset contains information about users on a Social Networking site and using that info as Features for our ML model,
we try to predict that whether a particular user after clicking on a ad on the Social networking site goes on to buy a particular product or not.
Well this particular Social Network has a Business client which lets assume is a car company which advertises itself by putting adds on the social networking site.
Now the work of the social network here is to gather information as to whether the user bought the product or not.
The dependent variable in this case is Purchased which is 1 if user purchases the car and 0 otherwise.
So the goal here is to create a classifier which would put each user into the correct category by predicting as to whether he's buying the product or not.

"""
dataset = pd.read_csv('Social_Network_Ads.csv')
print(dataset.head())
#The following features will be considered as the independent variables...
#...1)Age
#...2)Estimated Salary
#Now some of you might be wondering that the dataset also contains 3 more columns and why are we leaving them?
#Well the answer to that is quite simple...and we will soon see the reason as to why each of them is being dropped.
#...1)UserId- The UserId has no effect on whether the user would purchase the Car or not
#...2)Gender- Some might say that Gender would play a role but that is really subjective to discuss.
#Moreover Since gender is a Categorical variable we would have to use Variabale Encoder for it.
X = dataset.iloc[:, [2, 3]].values
#Storing the dependent variable in y i.e. Purchased which is 1 if user purchases the car and 0 otherwise.
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
#Importing the Cross Validation library which is now known as ModelSelection in newer versions of Python
from sklearn.model_selection import train_test_split
#Splitting the dataset into training set and testing set
#We divide the data into 75% data for training and 25% for testing our data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


'''
Now are we going to apply Feature Scaling?
Yes, Definitely we will be applying feature scaling because we want accurate prediction i.e. we want to predict which users are going to buy the car or not.

'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
#Creating the standard Scalar Object of the Preprocessing Class
sc = StandardScaler()
#scaling X_train by fitting the Standard Scalar object to our Matrix of Features X_train
X_train = sc.fit_transform(X_train)
#scaling X_test in the same basis
X_test = sc.transform(X_test)
#To actually see the difference and confirm that they are almost upto the same scale,if you want you can...
print(X_train)
print(X_test)

# Fitting K-NN to the Training set
#So we need to import the scikit.neighbours library and from it we would import the KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
#creating an object of the class
#Inspect the classifier by pressing Ctrl+Q to show the Documentation and seeing all the parameters with their def accordingly
#No of nearest neighbours=5(Default)
#Specify meteric as 'minkowski' and power as '2' for using the Eucledian Distance for KNN==> set p=2
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#Now we fit the classifier object to our training set
classifier.fit(X_train, y_train)

# Predicting the Test set results
#Since the classifier has been fit to the Dataset we can predict the Outcomes of the test set.
y_pred = classifier.predict(X_test)
#Displaying out the predicted values
print(y_pred)
c=0
for i in range(0,len(y_pred)):
    if(y_pred[i]==y_test[i]):
        c=c+1
accuracy=c/len(y_pred)
print("Accuracy is")
print(accuracy)
#As you can see that the accuracy turned out to be 93% which is great achievement for our classifier.
#----------------------------------------------------------------------
#DATA VISUALIZAION AND CONFUSION MATRIX
#Now we come to a much more interesting part in which we try to find out as to whether the prediciton boundary
#is a Linear or a Non Linear Classifier...It would be Linear if the separator line is a straight line and non linear if its curved.

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Training set results
#As it might be visible in the newly opened graph of the Training Set that we have a Non-Linear Classifier which fits the Data pretty well.
#Well apart from the very few misclassified points...Red points in Green region or vice versa our Model does a pretty decent job in classifying these points.

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
#Just as we had performed an Analysis on the Graph of the Training set before we now perform one on the Test set results...
#Yet Again we see that most of the points are correctly classified just with a few exceptions which is fine by the way to have,
#because we are trying to prevent our model from Overfitting which we know can be a serious threat.
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()