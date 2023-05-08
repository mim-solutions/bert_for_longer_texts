# Installing dependencies

On top of project dependencies, install additional doc dependencies

    pip install -r sphinx-docs/requirements.txt

# Testing locally

1. Generate docs `sphinx-build sphinx-docs docs`
2. Start a simple http server `python -m http.server --directory docs`
3. Open http://0.0.0.0:8000/

# Publishing

1. Make sure you are on the `gh-pages` branch.
2. Generate docs `sphinx-build sphinx-docs docs`
3. Add and commit `docs`
4. Push changes.
