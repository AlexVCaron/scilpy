#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile
import pytest

from scilpy.io.fetcher import get_testing_files_dict, fetch_data    
from scilpy import SCILPY_HOME

tmp_dir = tempfile.TemporaryDirectory()

fetch_data(get_testing_files_dict(), keys=['tracking.zip'])

def test_run(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'tracking', 'seeding_mask.nii.gz')
    ret = script_runner.run('scil_viz_fodf.py', in_fodf,
                            '--silent',
                            '--in_transparency_mask', in_mask,
                            "--output", "fodf.png")
    assert ret.success, ret.stderr
