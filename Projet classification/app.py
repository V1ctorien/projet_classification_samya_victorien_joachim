from flask import Flask, redirect , render_template, request, flash, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pandas as pd
import joblib   
from werkzeug.utils import secure_filename
import sqlite3
import numpy as np
from model_test.fraud import FraudDetectorFunctions as ff
from model_test.EDA import olap , matrice_confusion, matrice_correlation, fraudpartype


app = Flask(__name__)
model = joblib.load('model_test/rf_model.pkl')

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

fraud_detector = ff(
    model=model,
    file_path=r"data\data_train_set.csv"
)

@app.route("/")
def home():
    return render_template("home.html")

app.secret_key = os.urandom(24)  # Clé secrète pour sécuriser les sessions

# Simulation d'une base de données (en pratique, utiliser une vraie DB)
users_db = {
    "admin": generate_password_hash("1234"),  # utilisateur: mot de passe
    "user": generate_password_hash("password")
}


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        # Validation des champs
        if not username or not password:
            flash("Veuillez remplir tous les champs.", "error")
            return redirect(url_for('login'))

        # Vérification des identifiants
        if username in users_db and check_password_hash(users_db[username], password):
            session['username'] = username
            flash("Connexion réussie !", "success")
            return redirect(url_for('upload'))
        else:
            flash("Nom d'utilisateur ou mot de passe incorrect.", "error")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    #  Vérifie si l'utilisateur est connecté
    rt=None
    y_pred=None
    stat_gen=ff(model=model, file_path=r"data\card_credit_fraud_Classification project Final.csv_sans_guillemets.csv").afficher_stat_gen(file_path=r"data\card_credit_fraud_Classification project Final.csv_sans_guillemets.csv")
    if 'username' not in session:
        flash("Veuillez vous connecter d'abord.", "error")
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'upload_file':
            try :
                
                # Vérifie si un fichier a été envoyé
                if 'file' not in request.files:
                    flash("Aucun fichier sélectionné.", "charge_error")
                    return redirect(request.url)

                file = request.files['file']

                if file.filename == '':
                    flash("Aucun fichier sélectionné.", "charge_error")
                    return redirect(request.url)

                # Vérifie que c'est un CSV
                if not file.filename.endswith('.csv'):
                    flash("Seuls les fichiers CSV sont autorisés.", "charge_error")
                    return redirect(request.url)
                    
                # Sécurise le nom du fichier
                filename = secure_filename(file.filename)

                        # Chemin COMPLET vers le CSV
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename) 

                        # Sauvegarde du fichier
                file.save(filepath) #
                df = ff(model, filepath)
                rt=df.charger_transactions()
                print("RT calculé :", rt)

                
                    
                flash(f"Fichier '{file.filename}' chargé avec succès !", "charge_error")

            except Exception as e:
                    flash(f"Erreur : {e}", "charge_error")
                    return redirect(request.url)    
            
        elif action == 'predict_id':
            try:

                data_test=ff(model=model,file_path="uploads\data_test_set.csv")
                data_test.setindex()
                
                
                transID=int(request.form.get('transactionId'))
                # transID_str = request.form[]
                # transID = int(transID_str)
                
                
                                  
                if transID not in data_test.dataset["transactionId"].values :
                    flash("ID non valide","predict_error")
                    y_pred=None

                else:
                    y_pred=data_test.saisie_transaction(transID)
            except Exception as e:
                        flash(f"Erreur : {e}", "predict_error")
                        return redirect(request.url)
    
    print("y_pred =", y_pred)
    print("type(y_pred) =", type(y_pred))    
    return render_template('upload.html',stat_gen=stat_gen, y_pred=y_pred,rt=rt)

# @app.route('/eda', methods=['GET'])
# def eda():
#     olap_= olap()
#     fraudpartype_= fraudpartype()
#     matrice_correlation_= matrice_correlation()
#     matrice_confusion_= matrice_confusion("data/data_train_set.csv")
#     return render_template('eda.html', fraudpartype=fraudpartype_, matrice_correlation=matrice_correlation_, matrice_confusion=matrice_confusion_, olap=olap_)

@app.route('/eda', methods=['GET'])
def eda():
    olap_img = olap()
    fraud_type_img = fraudpartype()
    corr_img = matrice_correlation()
    conf_img = matrice_confusion("data/data_train_set.csv")

    return render_template(
        'eda.html',
        fraudpartype=fraud_type_img,
        matrice_correlation=corr_img,
        matrice_confusion=conf_img,
        olap=olap_img
    )



# @app.route('/logout')
# def logout():
#     session.pop('username', None)
#     flash("Vous êtes déconnecté.", "info")
#     return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=False)