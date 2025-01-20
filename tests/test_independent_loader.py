from unittest import TestCase
from dataloader.independent_loader import *


from database import init_engine
from database.fetch import Fetch


class TestInDependentLoader(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestInDependentLoader, self).__init__(*args, **kwargs)
        init_engine()
        self.fetch = Fetch()

    def test_load_participation_features(self):
        p = self.fetch.fetch_participation.one(horse_id="J064", race_id="2025:1")
        features = load_participation_features(p)
        self.assertEqual(features, [9, 10])
