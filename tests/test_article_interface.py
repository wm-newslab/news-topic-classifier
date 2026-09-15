import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import localnews_classifier as api


class ArticleInterfaceTests(unittest.TestCase):
    def test_real_extraction_from_html(self):
        html = b'''<html><head><title>City opens a new bus route</title></head>
        <body><nav>Home Subscribe Contact</nav><article><h1>City opens a new bus route</h1>
        <p>The city transit authority announced a new bus route connecting downtown with residential neighborhoods. Service will begin Monday and buses will run every twenty minutes throughout the day.</p>
        <p>The route will serve the hospital and public library. Transit officials said the expansion would help residents reach essential services without driving. New stops include accessible boarding platforms.</p>
        </article></body></html>'''
        response = Mock()
        response.headers.get_content_type.return_value = 'text/html'
        response.read.return_value = html
        response.geturl.return_value = 'https://example.com/article'
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch.object(api, 'urlopen', return_value=response):
            result = api.extract_article('https://example.com/article')
        self.assertIn('bus route', result['title'])
        self.assertIn('twenty minutes', result['content'])
        self.assertNotIn('Home Subscribe Contact', result['content'])

    def test_invalid_url(self):
        with self.assertRaises(ValueError):
            api.extract_article('file:///tmp/article')

    def test_api_preserves_saved_settings_and_reuses_runtime(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            config = dict(base_model='base', max_input_chars=1500, max_seq_length=2048,
                          prompt_variant='full', truncation_strategy='head_tail', experiment_id='test')
            (Path(tmp)/'configuration.json').write_text(json.dumps(config))
            classifier = api.LocalNewsClassifier(Path(tmp)/'adapter')
            backend = Mock()
            backend.classify_dataframe.return_value = pd.DataFrame([dict(
                title='Title', content='Body', predicted_label='Transportation',
                raw_model_output='Transportation', classification_status='classified')])
            classifier._runtime = (backend, object(), object(), object())
            for _ in range(2):
                result = api.classify_article('Title', 'Body', classifier=classifier)
                self.assertEqual(result['predicted_label'], 'Transportation')
            self.assertEqual(backend.classify_dataframe.call_count, 2)
            self.assertEqual(backend.classify_dataframe.call_args.kwargs['max_input_chars'], 1500)
            with self.assertRaises(ValueError):
                classifier.classify('Title', '  ')

    def test_content_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            body = Path(tmp) / "article.txt"
            body.write_text("Full article body", encoding="utf-8")
            with patch.object(api, 'LocalNewsClassifier') as cls:
                cls.return_value.classify.return_value = {'classification_status': 'classified'}
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(api.main(['--title', 'Title', '--content-file', str(body)]), 0)
                cls.return_value.classify.assert_called_once_with('Title', 'Full article body')

    def test_cli_modes_and_errors(self):
        result = {'predicted_label': 'Health', 'classification_status': 'classified'}
        with patch.object(api, 'LocalNewsClassifier') as cls:
            cls.return_value.classify.return_value = result
            cls.return_value.classify_url.return_value = result
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(api.main(['--title', 'Title', '--content', 'Body']), 0)
            self.assertEqual(json.loads(out.getvalue()), result)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(api.main(['--url', 'https://example.com']), 0)
            cls.return_value.classify_url.assert_called_once_with('https://example.com', timeout=30)
            cls.return_value.classify_url.side_effect = ValueError('No article body')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(api.main(['--url', 'https://example.com']), 1)
                with self.assertRaises(SystemExit):
                    api.main(['--url', 'https://example.com', '--content', 'Body'])


if __name__ == '__main__':
    unittest.main()
