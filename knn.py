from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import datasets
iris=datasets.load_iris()
iris_data=iris.data
iris_labels=iris.target
x1,x2,y1,y2=train_test_split(iris_data,iris_labels,test_size=0.20)
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(x1,y1)
y_pred=classifier.predict(x2)
print('Confusion Matrix is as follows')
print(confusion_matrix(y2,y_pred))
print('Accuracy Matrices')
print(classification_report(y2,y_pred))