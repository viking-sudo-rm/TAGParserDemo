# TAGParserDemo

Web demo for graph-based TAG parsing. See https://github.com/jungokasai/graph_parser.

## Some commands

If static files are missing:

```bash
python manage.py collectstatic
```

To restart the server:

```bash
sudo apachectl restart
```
If tokenizers are missing for NLTK, run the following in the Python shell:

```python
import nltk
nltk.download("punkt")
```
