import hashlib
import subprocess
import sys

import pytest
import yaml

from robometrics.datasets import (
    demo_raw,
    list_problem_groups,
    load_raw_dataset,
    motion_benchmaker_raw,
    mpinets_raw,
    structure_problems,
)
from robometrics.file_structure import get_dataset_path


@pytest.mark.parametrize(
    "name,legacy_loader,group_count,problem_count,sha256",
    [
        ("demo", demo_raw, 1, 5,
         "712b9fc93802cc39ed769b32b1e3ad5006196db73962da59ebd68012c44ce74d"),
        ("mb", motion_benchmaker_raw, 8, 800,
         "5165ad4fb55f93c63dbbdda6f25a14f3ceb24ee3305e27ad317f38f1e0d6ed0d"),
        ("mpinets", mpinets_raw, 12, 1800,
         "9189186d83e51600a2c768aa7657933aa052150501852771994757052b797bbf"),
    ],
)
def test_packaged_data_matches_baseline(
    name, legacy_loader, group_count, problem_count, sha256
):
    source = (get_dataset_path() / (name + "_set.yaml")).read_bytes()
    assert hashlib.sha256(source).hexdigest() == sha256
    # The original reader used safe_load. Compare every field and problem order,
    # including any numerical differences introduced by the faster C loader.
    reference = yaml.safe_load(source)
    assert load_raw_dataset(name) == legacy_loader() == reference
    sizes = list_problem_groups(name)
    assert list(sizes) == list(reference)
    assert sizes == {group: len(problems) for group, problems in reference.items()}
    assert len(sizes) == group_count
    assert sum(sizes.values()) == problem_count
    if name == "mb":
        assert set(sizes.values()) == {100}


def test_explicit_data_directory(tmp_path):
    (tmp_path / "demo_set.yaml").write_text("custom_group: [{start: [1, 2]}]\n")
    expected = {"custom_group": [{"start": [1, 2]}]}
    assert load_raw_dataset("demo", tmp_path) == expected
    assert load_raw_dataset("demo", str(tmp_path)) == expected
    assert list_problem_groups("demo", tmp_path) == {"custom_group": 1}


def test_reads_do_not_share_mutable_data():
    first = demo_raw()
    second = load_raw_dataset("demo")
    first["bookshelf_small_panda"][0]["start"][0] = 999
    assert second["bookshelf_small_panda"][0]["start"][0] != 999
    assert demo_raw() == second


@pytest.mark.parametrize("reader", [load_raw_dataset, list_problem_groups])
def test_unknown_dataset(reader, tmp_path):
    with pytest.raises(ValueError, match="unknown dataset.*choose one of"):
        reader("unknown", tmp_path)


def test_missing_file_reports_location(tmp_path):
    with pytest.raises(FileNotFoundError, match="set data_dir") as error:
        load_raw_dataset("mb", tmp_path)
    assert str(tmp_path / "mb_set.yaml") in str(error.value)


@pytest.mark.parametrize("text", ["", "- problem\n", "42\n"])
def test_non_mapping_yaml(tmp_path, text):
    (tmp_path / "demo_set.yaml").write_text(text)
    with pytest.raises(ValueError, match="must contain a mapping"):
        load_raw_dataset("demo", tmp_path)


def test_safe_loader_fallback(tmp_path, monkeypatch):
    monkeypatch.delattr(yaml, "CSafeLoader", raising=False)
    (tmp_path / "demo_set.yaml").write_text("custom_group: []\n")
    assert load_raw_dataset("demo", tmp_path) == {"custom_group": []}


def test_raw_loading_without_numerical_or_geometry_dependencies():
    code = """
import importlib.abc
import sys

class BlockDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'geometrout', 'urchin', 'jax', 'torch', 'curobo', 'planigo'}:
            raise ImportError('unexpected dependency: ' + fullname)

sys.meta_path.insert(0, BlockDependencies())
from robometrics.datasets import demo_raw, list_problem_groups
assert len(demo_raw()['bookshelf_small_panda']) == 5
assert list_problem_groups('demo') == {'bookshelf_small_panda': 5}
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_structured_data_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "geometrout", None)
    with pytest.raises(AssertionError, match=r"pip install 'geometrout>=0\.1\.0\.10,<0\.2'"):
        structure_problems({})
