# Devoir - CNN "from scratch" vs Transfer Learning (Cats vs Dogs)

**Etudiant :** Karifa Kouyaté
**Formation :** Master 1 IA DIT
**Date :** Septembre 2026
**Enseignant :** M. Diallo (diallomous@gmail.com)

## Objectif du projet

Ce projet compare deux approches de reseaux de neurones convolutifs (CNN) pour un probleme de classification binaire (chats vs chiens) :

1. Experience A (CNN from scratch) : architecture personnalisee a 3 blocs convolutifs, avec regularisation par Batch Normalization et Dropout, optimisee avec Adam.
2. Experience B (Transfer Learning) : architecture pre-entrainee ResNet18, dont les couches de base sont gelees, avec une nouvelle couche de classification finale, optimisee avec SGD.
3. Experience C (comparaison d'optimiseurs) : la meme architecture from scratch que l'Experience A, reentrainee avec SGD, afin de comparer deux optimiseurs a configuration egale.

L'etude met en evidence l'impact de ces approches sur la vitesse de convergence, la performance finale et la capacite de generalisation sur des donnees inedites.

## Configuration de l'environnement

Le projet a ete developpe et execute sur macOS (MacBook Pro) en exploitant l'acceleration materielle de la puce Apple Silicon via le peripherique MPS (Metal Performance Shaders).

### Installation des dependances

```bash
# Creation et activation d'un environnement virtuel (optionnel)
python3 -m venv env
source env/bin/activate

# Installation des dependances requises
pip install -r requirements.txt
```

Contenu du fichier requirements.txt : torch, torchvision, matplotlib, numpy, scikit-learn, jupyter.

## Organisation des donnees

Le jeu de donnees utilise provient du corpus officiel Cats vs Dogs de Kaggle. Les donnees ne sont pas suivies par Git (voir .gitignore) et doivent etre telechargees puis placees localement selon l'arborescence suivante :

```
cnn-catsdogs-KouyateKarifa/
├─ Cat_Dog_data/
│  ├─ train/
│  │  ├─ cat/
│  │  └─ dog/
│  ├─ test/
│  │  ├─ cat/
│  │  └─ dog/
├─ notebook.ipynb
├─ .gitignore
├─ requirements.txt
└─ README.md
```

Les transformations appliquees incluent de la data augmentation sur le jeu d'entrainement (RandomRotation, RandomHorizontalFlip, RandomResizedCrop) afin d'ameliorer la robustesse des modeles et de limiter le surapprentissage. Un split train/validation de 80/20 est realise a l'aide de SubsetRandomSampler.

## Commandes et execution

L'integralite du pipeline est centralisee dans le notebook, a executer du haut vers le bas.

### 1. CNN from scratch avec Adam (Experience A)

* Architecture : 3 blocs Conv2d puis BatchNorm2d puis ReLU puis MaxPool2d, suivis de deux couches lineaires avec Dropout(0.5).
* Optimiseur : Adam (lr = 0.001).
* Fonction de perte : CrossEntropyLoss.
* Reproductibilite : seed fixe a 42.
* Sauvegarde : best_model_scratch.pth.

### 2. Transfer Learning avec SGD (Experience B)

* Architecture : base ResNet18 dont toutes les couches d'extraction de caracteristiques sont gelees (requires_grad = False). Couche finale remplacee par Linear(512, 2).
* Optimiseur : SGD (lr = 0.001, momentum = 0.9) applique uniquement sur la couche finale.
* Sauvegarde : best_model_transfer.pth.

### 3. CNN from scratch avec SGD (Experience C)

* Meme architecture que l'Experience A, meme seed, memes donnees.
* Optimiseur : SGD (lr = 0.01, momentum = 0.9).
* But : comparer Adam et SGD a configuration egale sur la meme architecture.
* Sauvegarde : best_model_scratch_sgd.pth.

### 4. Reimportation locale et evaluation finale (point obligatoire 6)

Le meilleur modele de chaque experience (selon la val loss) est sauvegarde automatiquement. La derniere cellule du notebook recharge ces fichiers, evalue les modeles sur le jeu de test, calcule accuracy, precision et recall, et affiche les matrices de confusion.

## Resultats et analyse comparative

### Metriques du meilleur modele en validation

| Modele / Experience | Val Loss | Accuracy (Val) | Precision (Val) | Recall (Val) |
| --- | --- | --- | --- | --- |
| From Scratch (Adam) | 0.6071 | 67.56% | 0.7319 | 0.5481 |
| From Scratch (SGD) | a completer | a completer | a completer | a completer |
| Transfer Learning (SGD) | 0.0555 | 97.89% | 0.9689 | 0.9893 |

### Metriques finales sur le jeu de test

Ces valeurs sont produites par la derniere cellule du notebook. A voir après exécution de la cellule.

| Modele / Experience | Accuracy (Test) | Precision (Test) | Recall (Test) |
| --- | --- | --- | --- |
| From Scratch (Adam) | a completer | a completer | a completer |
| From Scratch (SGD) | a completer | a completer | a completer |
| Transfer Learning (SGD) | a completer | a completer | a completer |

### Analyse comparative

L'analyse des courbes met en evidence une superiorite nette et immediate de l'approche par Transfer Learning (ResNet18) par rapport aux modeles entraines from scratch. Des la premiere epoque, le modele pre-entraine atteint une exactitude en validation superieure a 97%, la ou le modele personnalise se stabilise autour de 65 a 68% apres cinq epoques. Cela s'explique par le fait que ResNet18 dispose deja de filtres visuels complexes et optimises (formes, textures, contours) appris en amont sur le jeu ImageNet.

La comparaison des optimiseurs (Experience A avec Adam contre Experience C avec SGD), realisee a configuration egale sur la meme architecture, permet d'isoler l'effet du choix de l'optimiseur sur la convergence. Adam adapte automatiquement le pas d'apprentissage par parametre et converge en general plus vite en debut d'entrainement, tandis que SGD avec momentum peut offrir une meilleure generalisation lorsque le learning rate est correctement regle. Les valeurs a reporter dans les tableaux ci-dessus permettent de trancher pour ce jeu de donnees precis.

L'introduction conjointe de la Batch Normalization et du Dropout (0.5) sur le modele from scratch reste indispensable : sans ces mecanismes de regularisation, l'ecart entre la perte d'entrainement et la perte de validation se creuserait rapidement, signe d'un surapprentissage provoque par la dimensionnalite elevee des images d'entree (224x224). Le Transfer Learning, en gelant les poids profonds de l'extracteur, agit lui-meme comme un regularisateur, ce qui explique ses excellentes performances de generalisation. On note egalement que le modele from scratch entraine avec Adam presente un recall en validation nettement plus faible que sa precision (0.55 contre 0.73), ce qui indique qu'il manque une part importante d'exemples de la classe positive.

## Limites et pistes d'amelioration

* Limites actuelles : le nombre d'epoques a ete volontairement limite a 5 pour des raisons de temps de calcul. Les modeles from scratch n'ont pas atteint leur plein potentiel de convergence.
* Pistes d'amelioration :
  1. Fine-tuning progressif des dernieres couches convolutives de ResNet18, en plus de la couche de classification, pour adapter plus finement l'extraction de caracteristiques.
  2. Ajout d'un planificateur de taux d'apprentissage (Learning Rate Scheduler comme StepLR ou CosineAnnealingLR) pour affiner la descente de gradient en fin d'entrainement.
  3. Recherche plus systematique du learning rate pour chaque optimiseur (par exemple via un balayage de valeurs).
