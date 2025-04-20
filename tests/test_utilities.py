import pytest
import pandas as pd
from kaggle_nlp.utils.utilities import PreprocessingUtils

class TestPreprocessingUtils:
    def test_clean_keywords_nan_and_percent20(self):
        df = pd.DataFrame({'keyword': [None, 'fire%20alarm', 'earthquake']})
        cleaned = PreprocessingUtils.clean_keywords(df.copy())
        assert cleaned['keyword'][0] == 'unknown'
        assert cleaned['keyword'][1] == 'fire alarm'
        assert cleaned['keyword'][2] == 'earthquake'

    def test_combine_text(self):
        row = {'keyword': 'flood', 'text': 'River overflowed'}
        combined = PreprocessingUtils.combine_text(row)
        assert combined == 'keyword: flood text: River overflowed'

    @pytest.mark.parametrize('text,expected', [
        ('Check this out http://test.com', 'check this out '),
        ('@user Hello!', ' hello'),
        ('#disaster happened', 'disaster happened'),
        ('Special $%^&* chars!', 'special  chars'),
        ('MiXeD CaSe', 'mixed case'),
    ])
    def test_clean_text(self, text, expected):
        assert PreprocessingUtils.clean_text(text) == expected
