"""Check process isolation against the real installed wrapper and a tiny checkpoint."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from transformers import ModernBertModel
from test_reference import local_model
from reference import capture
from reference_process import capture_isolated


def embedding_checkpoint(path):
    local_model(path, ModernBertModel)
    (path/'modules.json').write_text(
        '[{"idx":0,"name":"0","path":"","type":"sentence_transformers.models.Transformer"},'
        '{"idx":1,"name":"1","path":"1_Pooling","type":"sentence_transformers.models.Pooling"}]')
    (path/'1_Pooling').mkdir()
    (path/'1_Pooling/config.json').write_text(
        '{"word_embedding_dimension":8,"pooling_mode_cls_token":false,'
        '"pooling_mode_mean_tokens":true,"include_prompt":true}')


class IsolatedReference(unittest.TestCase):
    def test_fresh_processes_preserve_native_and_mixed_batch_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root/'model'; model.mkdir()
            embedding_checkpoint(model)
            inputs = {'embedding': [
                {'id':'first','prefix':'','text':'query text','repeat':1},
                {'id':'second','prefix':'','text':'query','repeat':1}]}
            observed = {'schema':2,'batches':[{'id':'mixed/bucket',
                'keys':['first','second'],'ids':[[2,3],[2,0]],'mask':[[1,1],[1,0]],
                'conditions':['bucket']}]}
            expected = capture('embedding',model,observed,inputs)
            actual = capture_isolated('embedding',model,observed,inputs,root)
            self.assertEqual(actual,expected)
            events = [json.loads(line) for line in (root/'reference-events.jsonl').read_text().splitlines()]
            complete = [event for event in events if event['status']=='complete']
            self.assertEqual([(x['phase'],x['id'],x['shape']) for x in complete],
                             [('native','first',[1,2]),('native','second',[1,1]),('batch','mixed/bucket',[2,2])])
            self.assertEqual(len({x['pid'] for x in complete}),3)
            self.assertEqual([x['status'] for x in events],['started','inference','complete']*2+['started','complete'])

    def test_failed_unit_records_identity_and_does_not_start_the_next(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            inputs={'embedding':[{'id':'first','prefix':'','text':'query','repeat':1},
                                 {'id':'second','prefix':'','text':'text','repeat':1}]}
            with self.assertRaises(subprocess.CalledProcessError):
                capture_isolated('embedding',root/'missing',{'schema':2,'batches':[]},inputs,root)
            events=[json.loads(line) for line in (root/'reference-events.jsonl').read_text().splitlines()]
            self.assertEqual([(x['id'],x['status']) for x in events],
                             [('first','started'),('first','incomplete')])
            self.assertTrue((root/'reference-embedding-units/0.private.log').is_file())
            self.assertFalse((root/'reference-embedding-units/1.request.private.json').exists())


if __name__=='__main__':
    unittest.main()
