# Przydatne hooki do gita (trzeba ustawić w `.git/hooks`)

UWAGA! Trzeba ustawić u siebie lokalnie te pliki jako wykonywalne:

```
chmod a+x .git/hooks/pre-commit
```

### pre-commit

dodaje wersje "stripped" notebooków, które zawierają tylko kod pythona z tych notebooków
```
#!/bin/bash
git diff --cached --name-only --diff-filter=ACM | while IFS='' read -r line || [[ -n "$line" ]]; do
  if [[ $line == *.ipynb ]] ;
  then
    nb_dir=$(dirname $line)
    if [[ $nb_dir == "." ]]; then
        nb_dir=""
    fi
    filename=$(basename $line)
    stripped_dir=stripped/${nb_dir} #copy the directory structure
    mkdir -p $stripped_dir
    target_stripped_file="${stripped_dir}/${filename%.ipynb}_stripped.py"
    jupyter nbconvert --to script $line --output "$(git rev-parse --show-toplevel)/${target_stripped_file%.py}" #nbconvert blindly adds the suffix .py to the filename...
    sed -i 's/\\n/\n/g' $target_stripped_file
    git add $target_stripped_file
  fi
done
```