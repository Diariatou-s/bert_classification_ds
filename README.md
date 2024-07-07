# DS - Classification BERT

## Introduction

Ce repo contient une solution complète pour la classification de texte en utilisant BERT. Le modèle est entraîné pour classer les résumés trouvés sur Netflix, IMDb, Metacritic, et Rotten Tomatoes afin de déterminer si le contenu est un film ou une série. Le repo inclut :
1. Un script pour l'initialisation du dataset et l'entraînement d'un modèle de classification BERT.
2. Une interface utilisateur Gradio pour interagir avec le modèle.
3. Une API FastAPI pour fournir un service d'inférence du modèle.

## Contenu du repo

- `bert_classification.py` contient tout le code nécessaire pour charger le dataset, le prétraiter, et entraîner un modèle de classification BERT basé sur les résumés. Un fichier nommé `model_ds_netflix.pth` sera généré à la fin de l'éxécution de ce script
- `demo.py` permet de lancer une interface utilisateur Gradio qui facilite l'interaction avec le modèle précédemment entraîné et sauvegardé.

`![Gradio](images/gradio.JPG)`

- `api.py` met en place une API FastAPI qui permet d'interagir avec notre modèle via des requêtes HTTP, notamment une requête POST qui, à qui donné un texte, une prédiction du modèle est renvoyée.

`![Fast API](images/fastapi.JPG)`
