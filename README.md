# ANA-505-Week-6-Activity

#e1071 will be used for Support Vector Classification.
install.packages("e1071")
install.packages("GGally")
install.packages("ggplot2")
install.packages("caret")

library(e1071)
library(GGally)
library(ggplot2)

#get the dataset
data(iris)

#explore the data
str(iris)
head(iris,3)

#Create the SVM model
svm_model <- svm(Species ~ ., data=iris,
                 kernel="radial") #linear/polynomial/sigmoid

#Lets have a closer look at the parameters 
#and judge before hand if a good model can be created or not.
ggpairs(iris, ggplot2::aes(colour = Species, alpha = 0.4))

#We can clearly see from the Histograms of Petal.length 
#and Petal.width that we can clearly seperate out Setosa species with very high confidence.

#However, Versicolor and Virginica Species are overlapped. 
#If we look at the scatterplot of Sepal.Length vs Petal.Length 
#and Petal.Width vs Petal.Length, 
#we can distintly see a seperator that can be draw between the groups of Species.

#Looks like we can just use Petal.Width and Petal.Length as parameters 
#and come with a good model. SVM seems to be a very good model for this type of data.

plot(svm_model, data=iris,
     Petal.Width~Petal.Length,
     slice = list(Sepal.Width=3, Sepal.Length=4) 
)

#from the graph you can see data, support vector(represented by cross sign) 
#and decision boundry, belong to 3 types of species

#White color represented predicted class for second species(versicolor)

#Pink color represented predicted class for third species(virginica)

#Also we have 52 Support vector, 
#8 of them belongs to first species
#(You can see 8 cross in first class), 
#22 of them belongs to second species, 
#21 of them belongs to third species.

#Predict each Species
#Confusion matrix and missclassification error
pred = predict(svm_model,iris)
tab = table(Predicted=pred, Actual = iris$Species)
tab

#Get missclassification rate
1-sum(diag(tab)/sum(tab))

#How did the model do?

#Support vector machines (SVMs) are useful when there are very many input variables 
#or when input variables interact with the outcome or with each other in complicated (nonlinear) ways. 
#By observing the plots we can clearly see that some variables are non-linearly related to each other. 
#Hence, using SVM is a good option on the Iris dataset.

#Since in SVM we plot each data item as a point in n-dimensional space (where n is number of features you have) 
#with the value of each feature being the value of a particular coordinate and 
#then find a line that splits the data between two differently classified groups of data such that the 
#distances from the closest point in each of the two groups will be farthest away from this line drawn.
#Since our data is linearly seperable, SVM would be a good choice for classification purpose of Iris dataset which showed 
#accuracy of 97.3% which is highesr among all the methods


#What is the accuracy rate?

library(caret)

confusionMatrix(iris$Species, pred)

#Accuracy : 0.9733          
#95% CI : (0.9331, 0.9927)
#No Information Rate : 0.3333          
#P-Value [Acc > NIR] : < 2.2e-16       

#Kappa : 0.96  
