Application de Détection de Fraude

Cette application web permet de détecter des transactions frauduleuses à l’aide d’un modèle de machine learning.
Elle propose une interface web avec authentification, visualisation de données, prédictions automatiques et stockage en base de données.

---

-------- Prérequis

Si le projet est récupéré depuis GitHub, il est nécessaire de :

1 - Créer un environnement virtuel dans le dossier PROJET CLASSIFICATION
	python -m venv venv

	Activation :
	- Windows : venv\Scripts\activate
	- Linux / Mac : source venv/bin/activate
2 - Installer les dépendances
	pip install pandas scikit-learn matplotlib joblib werkzeug flask seaborn xgboost imblearn

---

----- Lancement de l’application

L’application se lance via le fichier app.py.

python app.py

Après le lancement, une adresse locale s’affiche dans la console.
Cliquez dessus pour ouvrir l’application dans votre navigateur.

---

--------- Authentification

- Identifiant : admin
- Mot de passe : 1234

 Les identifiants sont inscrits en dur pour le moment (usage test uniquement).

---

--------- Navigation dans l’application

 Page d’accueil
Après la connexion, vous arrivez sur la page principale.

 Dataviz
Un bouton Dataviz permet de :
- Visualiser les données du CSV principal
- Visualiser les données utilisées par le modèle

Vous pouvez remplacer le fichier CSV par un autre à condition qu’il ait le même nom et la même structure.

 Modèle de Machine Learning
- Le fichier creationmod.py permet de créer ou recréer un modèle
- Vous pouvez l’adapter pour entraîner un nouveau modèle

 Upload et prédictions
- Vous pouvez charger un fichier CSV (il sera sauvegardé dans le dossier upload/)
- Le CSV doit contenir plusieurs lignes et respecter la structure attendue
- A des fin de demonstatrions vous pouvez charger data_test_set.csv qui ce trouve dans le dossier /data
- Le programme sélectionne 15 lignes aléatoires :
  - 5 fraudes
  - 10 non-fraudes
- Les résultats sont :
  - Prédits par le modèle
  - Enregistrés dans une base de données

 Base de données
Deux tables sont automatiquement remplies :
- transaction
- compte
Les tables Client et utilisateur ne sont pas utilisé pour le moment. 

 Résultats affichés
Après la prédiction, l’application affiche :
- le nombre de lignes traitée
- Le nombre de lignes frauduleuses
- Le nombre de lignes non frauduleuses

 Prédiction individuelle
Vous pouvez entrer un ID de transaction pour obtenir le résultat :
- Fraude
- Non fraude

---

 Structure du projet 


data/
	.cvs

database/
	BDD

model_test/
	fraude.py (contient les fonctions de statistique et d'utilisation du model)
	EDA.py (contient les fonctions de visualisation)
	creationmod.py (entrainement du model)
	rf_model.pkl (model issue de l'entrainement utilisé pour la prediction)

static/
	images/
		.png
	.css

templates/
	.html

upload/

app.py
README.txt

---

 Remarques

- Projet à but pédagogique / démonstratif
- Sécurité (login, mot de passe) non adaptée à un usage en production
- Le format des fichiers CSV doit être strictement respecté
- pas de boutton de retour sur les pages
- pas de possibilitées d'enregistrement de nouvel utilisateur 
