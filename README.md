# Démarrer un projet MLflow avec uv et VS Code (scikit-learn + Iris)

Ce guide vous accompagne pas à pas, de la création du projet jusqu'à la visualisation des résultats dans **MLflow UI**, en travaillant dans un **notebook** ouvert dans **VS Code**.

**Ce que vous allez faire :**
- créer un projet Python avec `uv`
- entraîner plusieurs modèles sur le jeu de données Iris dans un notebook
- enregistrer paramètres, métriques et modèles avec MLflow
- comparer les essais dans l'interface web

---

## 0. Prérequis

- Un terminal (Linux, macOS ou Windows PowerShell)
- [VS Code](https://code.visualstudio.com/) installé
- Une connexion internet (pour installer les paquets)

Inutile d'installer Python à l'avance : `uv` peut le télécharger au besoin.

---

## 1. Installer uv

**Linux / macOS**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Fermez puis rouvrez le terminal, et vérifiez :

```bash
uv --version
```

> **Pourquoi uv ?** C'est un outil qui gère à la fois l'environnement virtuel, les dépendances et la version de Python, avec une seule commande et très rapidement.

---

## 2. Créer le projet

```bash
uv init mlflow-iris
cd mlflow-iris
```

`uv` génère automatiquement :

| Fichier | Rôle |
|---|---|
| `pyproject.toml` | description du projet et de ses dépendances |
| `.python-version` | version de Python utilisée |
| `main.py` | script d'exemple (nous ne l'utiliserons pas) |
| `README.md` | README du projet (à remplacer par ce guide si vous le souhaitez) |

---

## 3. Ajouter les dépendances

```bash
uv add mlflow scikit-learn
uv add --dev ipykernel
```

- La première commande installe `mlflow` et `scikit-learn`, et crée l'environnement virtuel `.venv/` s'il n'existe pas.
- La seconde installe `ipykernel`, le moteur qui permet à VS Code d'exécuter les cellules du notebook. L'option `--dev` indique que c'est un outil de développement, pas une dépendance du projet lui-même.

Les versions exactes sont notées dans `pyproject.toml` et `uv.lock` (reproductibilité).

---

## 4. Préparer VS Code

### 4.1 Installer l'extension

Dans VS Code, ouvrez l'onglet **Extensions** (`Ctrl+Maj+X`, ou `Cmd+Maj+X` sur macOS) et installez :

- **Jupyter** (éditeur Microsoft)
- **Python** (éditeur Microsoft, généralement installée avec la précédente)

### 4.2 Ouvrir le projet

Menu **Fichier > Ouvrir le dossier…** puis choisissez le dossier `mlflow-iris`.

> **Important :** ouvrez bien le dossier du projet (pas seulement un fichier), sinon VS Code ne trouvera pas l'environnement `.venv`.

---

## 5. Créer le notebook

1. Dans l'explorateur de VS Code, créez un fichier **`train.ipynb`** à la racine du projet.
2. Cliquez sur **Sélectionner le noyau** (en haut à droite du notebook).
3. Choisissez **Environnements Python…** puis l'entrée **`.venv`** du projet.

> **Pourquoi à la racine ?** Le notebook crée la base `mlflow.db` à côté de lui. En le gardant à la racine, la commande de l'étape 7 trouvera cette base sans réglage supplémentaire.

Un notebook complet, prêt à l'emploi, est fourni avec ce guide (`train.ipynb`). Vous pouvez aussi le recréer vous-même cellule par cellule grâce au tableau ci-dessous.

| N° | Type | Contenu |
|---|---|---|
| 1 | Code | Imports (`mlflow`, `scikit-learn`) |
| 2 | Code | Configuration MLflow : `set_tracking_uri` et `set_experiment` |
| 3 | Code | Chargement et découpage des données Iris |
| 4 | Code | Fonction `train_model()` qui crée un run MLflow |
| 5 | Code | Premier essai |
| 6 | Code | Autres essais avec d'autres hyperparamètres |

### La cellule la plus importante : la cellule 4

```python
def train_model(params, run_name=None):
    with mlflow.start_run(run_name=run_name):
        model = RandomForestClassifier(random_state=42, **params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="macro")

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1)
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"{params} -> accuracy={accuracy:.3f}, f1={f1:.3f}")
```

> **Pourquoi `skops_trusted_types` ?** MLflow enregistre les modèles scikit-learn dans un format sécurisé (`skops`) qui refuse par défaut certains objets internes, dont celui des forêts aléatoires (`sklearn.tree._tree.Tree`). Comme vous créez le modèle vous-même, vous pouvez le déclarer sûr. Ne faites jamais cela pour un modèle venant d'une source inconnue.

**Les 4 notions MLflow à retenir :**

| Notion | Rôle | Dans le notebook |
|---|---|---|
| **Experiment** | regroupe des essais liés | `set_experiment` |
| **Run** | un essai d'entraînement | `start_run` |
| **Params / Metrics** | réglages en entrée / résultats en sortie | `log_params`, `log_metric` |
| **Artifact** | fichier produit (ici le modèle) | `log_model` |

---

## 6. Exécuter le notebook

Exécutez les cellules **dans l'ordre** avec `Maj + Entrée`, ou cliquez sur **Tout exécuter** en haut du notebook.

Vous devez voir s'afficher une ligne par essai (paramètres, accuracy, f1). Deux éléments apparaissent alors à la racine du projet :

- `mlflow.db` : paramètres et métriques
- `mlruns/` : modèles enregistrés (artefacts)

---

## 7. Lancer MLflow UI

Ouvrez un terminal dans VS Code (menu **Terminal > Nouveau terminal**). Il s'ouvre dans le dossier du projet. Lancez :

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Ouvrez ensuite votre navigateur à l'adresse : **http://127.0.0.1:5000**

> **Important :** l'option `--backend-store-uri` doit pointer vers la même base que dans le notebook, sinon l'interface s'affichera vide.

> **Port 5000 déjà utilisé ?** (fréquent sur macOS à cause d'AirPlay) Ajoutez `--port 5001` et ouvrez http://127.0.0.1:5001.

Pour arrêter le serveur : `Ctrl + C` dans le terminal. Le serveur peut rester lancé pendant que vous relancez des cellules : actualisez simplement la page pour voir les nouveaux runs.

---

## 8. Explorer les résultats dans l'interface

1. Dans le menu de gauche, ouvrez l'expérience **iris-classification**.
2. Vous voyez un tableau avec un run par ligne : paramètres et métriques sont en colonnes.
3. Cochez les runs, puis cliquez sur **Compare** pour comparer graphiquement.
4. Cliquez sur un run pour consulter ses détails, puis l'onglet **Artifacts** pour voir le modèle enregistré.

---

## 9. Aller plus loin

**Recharger un modèle enregistré** : ajoutez une cellule au notebook (remplacez `<run_id>` par l'identifiant visible dans l'UI) :

```python
model = mlflow.pyfunc.load_model("runs:/<run_id>/model")
print(model.predict([[5.1, 3.5, 1.4, 0.2]]))
```

**Idées d'exercices :**
- ajouter un autre algorithme (par exemple `LogisticRegression`) et comparer
- enregistrer un graphique (matrice de confusion) avec `mlflow.log_figure`
- activer `mlflow.autolog()` pour laisser MLflow tout enregistrer automatiquement

---

## 10. Dépannage

| Problème | Piste |
|---|---|
| Le noyau `.venv` n'apparaît pas dans la liste | vérifier que `uv add --dev ipykernel` a été exécuté, puis recharger VS Code (`Ctrl+Maj+P` > *Developer: Reload Window*) |
| `ModuleNotFoundError: mlflow` dans le notebook | le mauvais noyau est sélectionné : choisir celui du `.venv` du projet |
| L'UI est vide | vérifier que `--backend-store-uri` correspond à `set_tracking_uri`, et que le terminal est dans le dossier du projet |
| `Address already in use` | changer de port avec `--port 5001` |
| `uv: command not found` | rouvrir le terminal après l'installation |
| Erreur d'exécution de script sous PowerShell | exécuter `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Erreur sur `log_model(name=...)` | mettre MLflow à jour (`uv add -U mlflow`) ou utiliser `artifact_path="model"` |

---

## 11. Fichier `.gitignore` conseillé

Si vous versionnez le projet avec Git, ajoutez :

```
.venv/
mlflow.db
mlruns/
```
