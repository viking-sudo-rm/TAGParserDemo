# TAGParserDemo

Web demo for graph-based TAG parsing. See https://github.com/jungokasai/graph_parser.

## Some commands

If static files are missing:

```bash
python manage.py collectstatic
```

To restart the server on Michelangelo:

```bash
sudo apachectl restart
```

To run a local Django server (for testing purposes only):

```bash
python manage.py runserver
```

If tokenizers are missing for NLTK, run the following in the Python shell:

```python
import nltk
nltk.download("punkt")
```
