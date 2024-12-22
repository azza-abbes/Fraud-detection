import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sympy.physics.quantum.matrixutils import sparse, np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

#train dataset:
df_train=pd.read_csv("C:/projML/fraudTrain/fraudTrain.csv")

#test dataset:
df_test=pd.read_csv("C:/projML/fraudTest/fraudTest.csv")

#verifier si le dataset est desequilibre:
print("\nAvant equilibrer les données: \n")
#pour le dataframe train :
non_fraud_train=df_train[df_train["is_fraud"]==0]
fraud_train=df_train[df_train["is_fraud"]==1]
print("nb de cas non_fraud dans train=",non_fraud_train.shape[0])
print("nb de cas fraud dans train=",fraud_train.shape[0])

#pour le dataframe test :
non_fraud_test=df_test[df_test["is_fraud"]==0]
fraud_test=df_test[df_test["is_fraud"]==1]
print("\nnb de cas non_fraud dans test=",non_fraud_test.shape[0])
print("nb de cas fraud dans test=",fraud_test.shape[0])

#visualisation pour les classes des dataframes:
plt.figure(figsize=(14, 6))

# Histogramme pour le dataframe Train
plt.subplot(1, 2, 1)
sns.histplot(df_train['is_fraud'], bins=2, kde=False, palette='viridis', color='blue')
plt.title("Distribution des classes dans le dataset Train", fontsize=14)
plt.xlabel("Classes", fontsize=12)
plt.ylabel("Nombre de cas", fontsize=12)
plt.xticks([0, 1], labels=["Non-Fraud", "Fraud"])

# Histogramme pour le dataframe Test
plt.subplot(1, 2, 2)
sns.histplot(df_test['is_fraud'], bins=2, kde=False, palette='viridis', color='green')
plt.title("Distribution des classes dans le dataset Test", fontsize=14)
plt.xlabel("Classes", fontsize=12)
plt.ylabel("Nombre de cas", fontsize=12)
plt.xticks([0, 1], labels=["Non-Fraud", "Fraud"])

plt.show()



# Equilibré les DataFrame:
print("\nAprés equilibrer les données: \n")
#dimunier le nbre des cas Non Fraud:
non_fraud_sample_train=non_fraud_train.sample(n=7506,random_state=42)
non_fraud_sample_test=non_fraud_test.sample(n=2145,random_state=42)

df_train_equilibre=pd.concat([non_fraud_sample_train,fraud_train])
df_test_equilibre=pd.concat([non_fraud_sample_test,fraud_test])

print("distribution de classe dans train: ",df_train_equilibre['is_fraud'].value_counts())
print("\ndistribution de classe dans test:",df_test_equilibre['is_fraud'].value_counts())

#visualisation pour les classes de dataframes equilibre:
plt.figure(figsize=(14, 6))

# Histogramme pour le dataframe Train
plt.subplot(1, 2, 1)
sns.histplot(df_train_equilibre['is_fraud'], bins=2, kde=False, palette='viridis', color='blue')
plt.title("Distribution des classes dans le dataset Train equilibré", fontsize=14)
plt.xlabel("Classes", fontsize=12)
plt.ylabel("Nombre de cas", fontsize=12)
plt.xticks([0, 1], labels=["Non-Fraud", "Fraud"])

# Histogramme pour le dataframe Test
plt.subplot(1, 2, 2)
sns.histplot(df_test_equilibre['is_fraud'], bins=2, kde=False, palette='viridis', color='green')
plt.title("Distribution des classes dans le dataset Test equilibré", fontsize=14)
plt.xlabel("Classes", fontsize=12)
plt.ylabel("Nombre de cas", fontsize=12)
plt.xticks([0, 1], labels=["Non-Fraud", "Fraud"])

plt.show()


# Visualisation des attributs par rapport à la classe is_fraud (dataset train équilibrée)


#  Répartition par profession (job) en fonction de la fraude
plt.figure(figsize=(10, 8))
top_jobs_eq = df_train_equilibre['job'].value_counts().head(10).index  # Top 10 professions
job_data_eq = df_train_equilibre[df_train_equilibre['job'].isin(top_jobs_eq)]
sns.countplot(y='job', hue='is_fraud', data=job_data_eq, palette='viridis')
plt.title('Top 10 professions en fonction de la fraude ', fontsize=14)
plt.xlabel('Nombre de cas', fontsize=12)
plt.ylabel('Profession', fontsize=12)
plt.legend(title='Fraude', labels=['Non-Fraud', 'Fraud'])
plt.show()

#  Répartition par ville (city) en fonction de la fraude
plt.figure(figsize=(10, 8))
top_cities_eq = df_train_equilibre['city'].value_counts().head(10).index  # Top 10 villes
city_data_eq = df_train_equilibre[df_train_equilibre['city'].isin(top_cities_eq)]
sns.countplot(y='city', hue='is_fraud', data=city_data_eq, palette='viridis')
plt.title('Top 10 villes en fonction de la fraude ', fontsize=14)
plt.xlabel('Nombre de cas', fontsize=12)
plt.ylabel('Ville', fontsize=12)
plt.legend(title='Fraude', labels=['Non-Fraud', 'Fraud'])
plt.show()

# Ajout d'une colonne transformée logarithmiquement pour visualiser le montant
"""df_train_equilibre['log_amount'] = np.log1p(df_train_equilibre['amt'])

plt.figure(figsize=(10, 6))
sns.histplot(data=df_train_equilibre, x='log_amount', hue='is_fraud', kde=True, bins=30, palette='coolwarm', element='step')
plt.title("Distribution Logarithmique de 'amount' par classe 'is_fraud'")
plt.xlabel("Log(1 + amount)")
plt.ylabel("Fréquence")
plt.show()"""



#split data :
#train:
X_train=df_train_equilibre.drop(columns=["is_fraud"])
y_train=df_train_equilibre['is_fraud']
#test :
X_test=df_test_equilibre.drop(columns=["is_fraud"])
y_test=df_test_equilibre['is_fraud']

#transformer String Data ==> Data numeriques **** TRAINING DATA *****:
#faire ces modifs sur des copies :
X_train_num=X_train.copy()

#trans_date_trans_time feature:
X_train_num['trans_date_trans_time'] = pd.to_datetime(X_train_num['trans_date_trans_time'])
X_train_num['hour'] = X_train_num['trans_date_trans_time'].dt.hour
X_train_num['day'] = X_train_num['trans_date_trans_time'].dt.day
X_train_num['month'] = X_train_num['trans_date_trans_time'].dt.month
X_train_num.drop(columns=['trans_date_trans_time'], inplace=True)

#date X_train_num of birth==>change it to age:
X_train_num['dob'] = pd.to_datetime(X_train_num['dob'])
X_train_num['age'] = (pd.to_datetime('today') - X_train_num['dob']).dt.days // 365
X_train_num.drop(columns=['dob'], inplace=True)

#drop first and last name, merchant ,trans_num features:
X_train_num.drop(columns=['merchant','first', 'last','trans_num'], inplace=True)

# tansformer String Data ==> Data numeriques *** TEST DATA ****:
X_test_num=X_test.copy()

#trans_date_trans_time feature:
X_test_num['trans_date_trans_time'] = pd.to_datetime(X_test_num['trans_date_trans_time'])
X_test_num['hour'] = X_test_num['trans_date_trans_time'].dt.hour
X_test_num['day'] = X_test_num['trans_date_trans_time'].dt.day
X_test_num['month'] = X_test_num['trans_date_trans_time'].dt.month
X_test_num.drop(columns=['trans_date_trans_time'], inplace=True, errors='ignore')

#date of birth==> change it to age:
X_test_num['dob'] = pd.to_datetime(X_test_num['dob'])
X_test_num['age'] = (pd.to_datetime('today') - X_test_num['dob']).dt.days // 365
X_test_num.drop(columns=['dob'], inplace=True)

#drop first and last name, merchant and trans_num features:
X_test_num.drop(columns=['merchant','first', 'last','trans_num'], inplace=True,errors='ignore')

#TEST et TRAIN data:
#category , gender , street, city, state, zip, job features in TEST et TRAIN:
encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
categorical_cols = [ 'category', 'gender', 'street', 'city', 'state', 'zip', 'job']
X_train_encoded_df = pd.DataFrame()
X_test_encoded_df = pd.DataFrame()
for col in categorical_cols:
    # Fit et transformation de data a trainer
    encoded_data_train = encoder.fit_transform(X_train_num[[col]])
    # recupereration les noms de features apres l'encodage
    feature_names = encoder.get_feature_names_out([col])
    # creation de dataframe a partir de la training data encodé
    encoded_df_train = pd.DataFrame(encoded_data_train, columns=feature_names, index=X_train_num.index)
    X_train_encoded_df = pd.concat([X_train_encoded_df, encoded_df_train], axis=1)

    # Transformation de  test data en utilisant le meme encoder
    encoded_data_test = encoder.transform(X_test_num[[col]])
    # creation de dataframe pour la test data encodé
    encoded_df_test = pd.DataFrame(encoded_data_test, columns=feature_names, index=X_test_num.index)
    X_test_encoded_df = pd.concat([X_test_encoded_df, encoded_df_test], axis=1)

# suppression des colonnes categoriale originale et concatenation des features encodé
X_train_num = X_train_num.drop(columns=categorical_cols)
X_train_num = pd.concat([X_train_num, X_train_encoded_df], axis=1)

X_test_num = X_test_num.drop(columns=categorical_cols)
X_test_num = pd.concat([X_test_num, X_test_encoded_df], axis=1)



scaler = StandardScaler()
X_train_sc= scaler.fit_transform(X_train_num)
X_test_sc= scaler.transform(X_test_num)

#Modeles:
#SVM:
model=svm.SVC(kernel='rbf',random_state=42)
model.fit(X_train_sc,y_train)

y_pred_svm=model.predict(X_test_sc)
matrice_conf_svm=confusion_matrix(y_test,y_pred_svm)
"""rappel_svm=recall_score(y_test,y_pred_svm)
precision_svm=precision_score(y_test,y_pred_svm)
f1_svm=f1_score(y_test,y_pred_svm)"""
print("matrice de confusion (svm):\n",matrice_conf_svm)
"""print("Rappel (svm):",rappel_svm)
print("Precision (svm) : ",precision_svm)
print("F1 score (svm): ",f1_svm)"""
print(classification_report(y_test,y_pred_svm))
#visualisation du matrice
plt.figure(figsize=(6, 5))
sns.heatmap(matrice_conf_svm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])

plt.title('Matrice de confusion (SVM)')
plt.xlabel('Prédictions')
plt.ylabel('Véritables valeurs')
plt.show()

#Logistic Regression:
model_lr=LogisticRegression(random_state=42)
model_lr.fit(X_train_sc,y_train)

y_pred_lr=model_lr.predict(X_test_sc)

matrice_conf_log_reg=confusion_matrix(y_test,y_pred_lr)
"""rappel_log_reg=recall_score(y_test,y_pred_lr)
precision_log_reg=precision_score(y_test,y_pred_lr)
f1_log_reg=f1_score(y_test,y_pred_lr)"""
print("matrice de confusion (logistic regression):\n",matrice_conf_log_reg)
"""print("Rappel (logistic regression):",rappel_log_reg)
print("Precision (logistic regression): ",precision_log_reg)
print("F1 score (logistic regression): ",f1_log_reg)"""
print(classification_report(y_test,y_pred_lr))
#visualisation du matrice
plt.figure(figsize=(6, 5))
sns.heatmap(matrice_conf_log_reg, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])

plt.title('Matrice de confusion (Logistic Regression)')
plt.xlabel('Prédictions')
plt.ylabel('Véritables valeurs')
plt.show()

#random forest
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_sc,y_train)
y_pred_rf = clf.predict(X_test_sc)
clf.score(X_test_sc,y_test)
print(classification_report(y_test,y_pred_rf))
#matrice de conf
matrice_conf_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(matrice_conf_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Matrice de confusion (Random Forest)')
plt.xlabel('Prédictions')
plt.ylabel('Véritables valeurs')
plt.show()
#knn
knn_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_model.fit(X_train_sc, y_train)
y_pred_knn = knn_model.predict(X_test_sc)
knn_model.score(X_test_sc,y_test)
print(classification_report(y_test,y_pred_knn))
#matrice con knn
matrice_conf_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(matrice_conf_knn, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Matrice de confusion (KNN)')
plt.xlabel('Prédictions')
plt.ylabel('Véritables valeurs')
plt.show()


# Liste des modèles supervisés
models = {
    'SVM': svm.SVC(kernel='rbf', random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=3, metric='euclidean')
}

# Effectuer la validation croisée
cv_results = {}
for model_name, model in models.items():
    # Validation croisée (cv=4)
    scores = cross_val_score(model, X_train_sc, y_train, cv=4, scoring='f1')  # Utilise 'f1' comme score
    cv_results[model_name] = scores
    print(f"Validation croisée pour {model_name}:")
    print(f"Scores: {scores}")
    print(f"Score moyen: {scores.mean():.4f}")
    print(f"Écart-type: {scores.std():.4f}")
    print("-" * 50)

# Visualisation des résultats de la validation croisée
cv_df = pd.DataFrame({
    'Model': list(cv_results.keys()),
    'Mean F1 Score': [scores.mean() for scores in cv_results.values()],
    'Standard Deviation': [scores.std() for scores in cv_results.values()]
})

plt.figure(figsize=(8, 6))
sns.barplot(x='Mean F1 Score', y='Model', data=cv_df, palette='viridis', ci=None)
plt.title('Résultats de la validation croisée (F1 Score moyen)')
plt.xlabel('F1 Score moyen')
plt.ylabel('Modèle')
plt.show()
