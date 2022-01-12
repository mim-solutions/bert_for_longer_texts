# Useful git hooks (should be added in `.git/hooks`)

They must be set as executables:

```
chmod a+x .git/hooks/pre-commit
```

### pre-commit

this hook adds stripped version of notebooks with raw python code - this makes analysis diffs and code review easier:

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