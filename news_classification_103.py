from sklearn.datasets import fetch_20newsgroups
# sample_cate 指定需要下载哪几个主题类别的新闻数据
sample_cate = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med', 'rec.sport.baseball']
# 需要从网络上下载，受连接外网速度限制, 可能要耐心等待几分钟时间
newsgroups_train = fetch_20newsgroups(subset='train', categories=sample_cate, shuffle=True)
# 以上得到训练集，以下代码得到测试集
newsgroups_test = fetch_20newsgroups(subset='test', categories=sample_cate, shuffle=True)

print(len(newsgroups_train.data))
print(len(newsgroups_test.data))

import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

count_vectorizer = CountVectorizer(stop_words='english')
cv_news_train_vector = count_vectorizer.fit_transform(newsgroups_train.data)
print("news_train_vector.shape:",cv_news_train_vector.shape)
cv_news_test_vector = count_vectorizer.transform(newsgroups_test.data)
print("news_test_vector.shape:",cv_news_test_vector.shape)

svc_clf = svm.SVC(kernel="linear")
svc_clf.fit(cv_news_train_vector,newsgroups_train.target)
cv_svc_predict = svc_clf.predict(cv_news_test_vector)
print("Classification report for classifier %s:\n%s\n" % (svc_clf,metrics.classification_report(newsgroups_test.target,cv_svc_predict,target_names=newsgroups_test.target_names)))

knn_clf = KNeighborsClassifier()
knn_clf.fit(cv_news_train_vector,newsgroups_train.target)
cv_knn_predict = knn_clf.predict(cv_news_test_vector)
print("Classification report for classifier %s:\n%s\n" % (knn_clf,metrics.classification_report(newsgroups_test.target,cv_knn_predict,target_names=newsgroups_test.target_names)))

nb_clf = MultinomialNB()
nb_clf.fit(cv_news_train_vector,newsgroups_train.target)
cv_nb_predict = nb_clf.predict(cv_news_test_vector)
print("Classification report for classifier %s:\n%s\n" % (nb_clf,metrics.classification_report(newsgroups_test.target,cv_nb_predict,target_names=newsgroups_test.target_names)))

###################################################################

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_news_train_vector = tfidf_vectorizer.fit_transform(newsgroups_train.data)
print("news_train_vectorizer.shape",tfidf_news_train_vector.shape)
tfidf_news_test_vector = tfidf_vectorizer.transform(newsgroups_test.data)
print("news_test_vectorizer.shape",tfidf_news_train_vector.shape)

svc_clf2 = svm.SVC(kernel="linear")
svc_clf2.fit(tfidf_news_train_vector,newsgroups_train.target)
tfidf_svc_predict2 = svc_clf2.predict(tfidf_news_test_vector)
print("Classification report for classifier %s:\n%s\n" % (svc_clf2,metrics.classification_report(newsgroups_test.target,tfidf_svc_predict2,target_names=newsgroups_test.target_names)))

knn_clf = KNeighborsClassifier()
knn_clf.fit(tfidf_news_train_vector,newsgroups_train.target)
tfidf_knn_predict = knn_clf.predict(tfidf_news_test_vector)
print("Classification report for classifier %s:\n%s\n" % (knn_clf,metrics.classification_report(newsgroups_test.target,tfidf_knn_predict,target_names=newsgroups_test.target_names)))

nb_clf = MultinomialNB()
nb_clf.fit(tfidf_news_train_vector,newsgroups_train.target)
tfidf_nb_predict = nb_clf.predict(tfidf_news_test_vector)
print("Classification report for classifier %s:\n%s\n" % (nb_clf,metrics.classification_report(newsgroups_test.target,tfidf_nb_predict,target_names=newsgroups_test.target_names)))
