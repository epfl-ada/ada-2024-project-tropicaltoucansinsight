
# Your project name
This is a template repo for your project to help you organise and document your code better. 
Please use this structure for your project and document the installation, usage and structure as below.

## Quickstart

```bash
# clone project
git clone <project link>
cd <project repo>

# [OPTIONAL] create conda environment
conda create -n <env_name> python=3.11 or ...
conda activate <env_name>


# install requirements
pip install -r pip_requirements.txt
```



### How to use the library
Tell us how the code is arranged, any explanations goes here.



## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

---

File containing the detailed project proposal (up to 1000 words)
- Title
- Abstract: A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why?
- Research Questions: A list of research questions you would like to address during the project.
- Proposed additional datasets (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible.
- Methods
- Proposed timeline
- Organization within the team: A list of internal milestones up until project Milestone P3.
- Questions for TAs (optional): Add here any questions you have for us related to the proposed project.

# Plan de projet (test)

Voici le diagramme de Gantt pour le projet :

```mermaid
gantt
    title Diagramme de Gantt du Projet

    section Etape 1 : Recherche de littérature
    Lecture et sélection des articles        :done,     des1, 2024-11-01, 10d
    Rédaction de la section état de l'art    :active,   des2, after des1, 10d

    section Etape 2 : Formulation des hypothèses
    Identification des hypothèses            :         hyp1, after des2, 5d
    Rédaction de la section hypothèses       :         hyp2, after hyp1, 7d

    section Etape 3 : Analyse des données
    Préparation et nettoyage des données     :         data1, 2024-11-20, 10d
    Analyse exploratoire                     :         data2, after data1, 10d
    Modélisation et test des hypothèses      :         data3, after data2, 10d

    section Etape 4 : Rédaction et révisions
    Rédaction du mémoire                     :         redac1, after data3, 20d
    Révision et corrections                  :         redac2, after redac1, 10d

    section Etape 5 : Présentation finale
    Préparation de la présentation           :         pres1, after redac2, 5d
    Soutenance                               :         pres2, after pres1, 1d
```


# Flowchart test
```mermaid
flowchart TD
    A[Début] --> B[Étape 1 : Recherche]
    B --> C{Est-ce validé ?}
    C -- Oui --> D[Étape 2 : Analyse]
    C -- Non --> E[Réviser]
    E --> B
    D --> F[Fin]
```

