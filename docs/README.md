# Documentation

This directory contains the complete documentation for Obscuro, optimized for GitHub Pages.

## Structure

- `index.md` - Documentation homepage and overview
- `cli-reference.md` - Complete CLI command reference
- `api-reference.md` - REST API endpoint documentation
- `configuration.md` - All configuration options explained
- `python-api.md` - Python library API guide

## Viewing Locally

You can view the documentation locally using Jekyll:

```bash
# Install Jekyll
gem install bundler jekyll

# Serve the docs
cd docs
jekyll serve

# Open http://localhost:4000
```

## GitHub Pages

The documentation is automatically published to GitHub Pages when pushed to the main branch.

Configure GitHub Pages in your repository settings:
1. Go to Settings → Pages
2. Set Source to "Deploy from a branch"
3. Select branch: `main`
4. Select folder: `/docs`
5. Save

The documentation will be available at: `https://tfaehse.github.io/obscuro/`

## Contributing

When adding new features, please update the relevant documentation files:

- New CLI options → `cli-reference.md`
- New API endpoints → `api-reference.md`
- New config options → `configuration.md`
- New Python APIs → `python-api.md`

Keep all documentation up-to-date with code changes.
