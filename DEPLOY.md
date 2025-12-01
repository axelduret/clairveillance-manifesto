# Clairveillance - Site Web

Site web du manifeste de la Clairveillance, publié via GitHub Pages.

🌐 **URL du site** : https://axelduret.github.io/clairveillance-manifesto/

## Structure

```
clairveillance-manifesto/
├── index.md                  # Page d'accueil
├── docs/
│   ├── CLAIRVEILLANCE.md     # Manifeste complet
│   └── TECHNOLOGIE.md        # Étude de cas
├── _layouts/
│   └── default.html          # Template avec navigation
├── _config.yml               # Configuration Jekyll
├── 404.md                    # Page d'erreur personnalisée
└── Gemfile                   # Dépendances Ruby
```

## Développement Local

### Prérequis

- Ruby 2.7+
- Bundler

### Installation

```bash
# Installer les dépendances
bundle install

# Lancer le serveur local
bundle exec jekyll serve

# Accéder au site
# http://localhost:4000/clairveillance-manifesto/
```

### Build

```bash
bundle exec jekyll build
```

Les fichiers générés seront dans `_site/`.

## Déploiement GitHub Pages

### Activation

1. Aller dans **Settings** du repo
2. Section **Pages**
3. Source : **Deploy from a branch**
4. Branch : **main** / dossier **/ (root)**
5. Save

Le site sera disponible à : `https://axelduret.github.io/clairveillance-manifesto/`

### Mise à jour

Chaque commit sur `main` déclenche automatiquement un rebuild.

## Configuration

### Personnalisation

Éditer `_config.yml` pour modifier :

- Titre, description, auteur
- Thème Jekyll
- Navigation
- Google Analytics (optionnel)
- URL de base

### Thèmes disponibles

GitHub Pages supporte ces thèmes :

- `jekyll-theme-minimal` (actuel)
- `jekyll-theme-cayman`
- `jekyll-theme-slate`
- `jekyll-theme-architect`
- `jekyll-theme-tactile`
- [Liste complète](https://pages.github.com/themes/)

## Maintenance

### Ajouter une page

1. Créer `nouvelle-page.md` à la racine ou dans `docs/`
2. Ajouter le front matter :

```yaml
---
layout: default
title: Titre de la page
---
```

3. Ajouter à la navigation dans `_config.yml` si besoin

### Modifier le style

Éditer la section `<style>` dans `_layouts/default.html`

## Licence

Contenu distribué sous [licence Creative Commons BY-SA 4.0](/docs/LICENSE.md/)

Voir le fichier [LICENSE](LICENSE) pour les détails complets.

## Contact

**Auteur** : Axel Duret  
**GitHub** : [@axelduret](https://github.com/axelduret)
