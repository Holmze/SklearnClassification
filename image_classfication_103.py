import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB



# The digits dataset
digits = datasets.load_digits()
print(digits.DESCR)

print(digits.images[0])

plt.imshow(digits.images[0],cmap=plt.cm.gray_r,interpolation='nearest')

_, axes = plt.subplots(4,4)
image_and_labels = list(zip(digits.images,digits.target))

for ax,(image,label) in zip(axes[0,:],image_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image,cmap = plt.cm.gray_r,interpolation = 'nearest')
    ax.set_title('Training: %i' % label)

n_samples = len(digits.images)
data = digits.images.reshape((n_samples,-1))

svc_clf = svm.SVC(gamma=0.001)

knn_clf = KNeighborsClassifier()

nb_clf = MultinomialNB()

X_train, X_test, y_train, y_test = train_test_split(data,digits.target,test_size=0.5,shuffle=False)

svc_clf.fit(X_train, y_train)
knn_clf.fit(X_train, y_train)
nb_clf.fit(X_train,y_train)

svc_predicted = svc_clf.predict(X_test)
knn_predicted = knn_clf.predict(X_test)
nb_predicted = nb_clf.predict(X_test)

svc_image_and_predictions = list(zip(digits.images[n_samples // 2:],svc_predicted))
knn_image_and_predictions = list(zip(digits.images[n_samples // 2:],knn_predicted))
nb_image_and_predictions = list(zip(digits.images[n_samples // 2:],nb_predicted))

for ax, (image, svc_prediction) in zip(axes[1,:],svc_image_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    ax.set_title('Prediction: %i' % svc_prediction)

for ax, (image, knn_prediction) in zip(axes[2,:],knn_image_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    ax.set_title('Prediction: %i' % knn_prediction)

for ax, (image, nb_prediction) in zip(axes[3,:],nb_image_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    ax.set_title('Prediction: %i' % nb_prediction)

plt.show()

print("Classification report for classifier %s:\n%s\n"
    % (svc_clf,metrics.classification_report(y_test,svc_predicted)))

print("Classification report for classifier %s:\n%s\n"
    % (knn_clf,metrics.classification_report(y_test,knn_predicted)))

print("Classification report for classifier %s:\n%s\n"
    % (nb_clf,metrics.classification_report(y_test,nb_predicted)))
    