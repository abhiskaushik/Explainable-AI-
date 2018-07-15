from sklearn.datasets import load_boston
import sklearn.ensemble
import sklearn.model_selection
import numpy as np

boston = load_boston()

rf = sklearn.ensemble.RandomForestRegressor(n_estimators=1000)

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(boston.data, boston.target, train_size=0.80)

print(rf.fit(train, labels_train))

print('Random Forest MSError', np.mean((rf.predict(test) - labels_test) ** 2))

print('MSError when predicting the mean', np.mean((labels_train.mean() - labels_test) ** 2))

# get the categorical features from the data
categorical_features = np.argwhere(np.array([len(set(boston.data[:,x])) for x in range(boston.data.shape[1])]) <= 10).flatten()

print(categorical_features)

import lime
import lime.lime_tabular


# build the explainer
explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=boston.feature_names, class_names=['price'], categorical_features=categorical_features, verbose=True, mode='regression')

#print(explainer)
# select 25th test value and check the prediction from the explainer
test_value = 25
exp = explainer.explain_instance(test[test_value], rf.predict, num_features=5)
print(exp)

exp.show_in_notebook(show_table=True)
exp.save_to_file('temp.html')
print(exp.as_list())
