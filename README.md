# TAGParserDemo

Web demo for graph-based TAG parsing. See https://github.com/jungokasai/graph_parser.

## Some commands

For missing submodules:

```shell
git submodule init
git submodule update
```

If static files are missing:

```shell
python manage.py collectstatic
```

To restart the server on Michelangelo:

```shell
sudo apachectl restart
```

To run a local Django server (for testing purposes only):

```shell
python manage.py runserver
```

If tokenizers are missing for NLTK, run the following in the Python shell:

```python
import nltk
nltk.download("punkt")
```
