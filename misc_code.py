# test to see that no columns/rows have been mislabled
# print(X_CV)
# print(y_CV)
# filtered_rows = eyt1['Pheno_Disc_Env1'][eyt1['Pheno_Disc_Env1'].iloc[:, 0] == 'GID6932153']
# print(filtered_rows)
# print(eyt1['Geno_Env1'].loc[['GID6932153']])

# print(X_train)
# print(y_train)

# fit and evaluate SVM model
# svm = SVC()
# svm.fit(X_train_scaled, y_train)
# svm_pred = svm.predict(X_CV_scaled)
# print(classification_report(y_CV, svm_pred))

# # fit and evaluate XGBoost model
# xgb = XGBClassifier()
# xgb.fit(X_train_scaled, y_train)
# xgb_pred = xgb.predict(X_CV_scaled)
# print(classification_report(y_CV, xgb_pred))

# # fit and evaluate neural network
# nn = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
# nn.fit(X_train_scaled, y_train)
# nn_pred = nn.predict(X_CV_scaled)
# print(classification_report(y_CV, nn_pred))