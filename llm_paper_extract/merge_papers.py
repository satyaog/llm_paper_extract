import argparse
import os
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
import unicodedata
import urllib
from time import sleep
from typing import Iterable, List
from typing import Iterable, List, Tuple

from pydantic import BaseModel, ValidationError
from pygments import highlight
from pygments.formatters import TerminalTrueColorFormatter
from pygments.lexers.data import YamlLexer
import yaml

from . import ROOT_DIR
from .model import ExtractionResponse, PaperExtractions, empty_paperextractions
from .utils import str_normalize

_STRIP_RE = r"[a-zA-Z0-9].*[a-zA-Z0-9]"
_EDITOR = os.environ.get("VISUAL", os.environ.get("EDITOR", None))
_TMPDIR = tempfile.TemporaryDirectory()

_READER = os.environ.get("READER", None)
if platform.system() == "Darwin":       # macOS
    _READER = _READER or "open"
else:                                   # linux variants
    _READER = _READER or "xdg-open"


def _open_editor(_f:str):
    if not os.path.exists(_f):
        raise FileNotFoundError(_f)
    p = subprocess.Popen((*_EDITOR.split(" "), _f))
    while p.poll() is None:
        sleep(1)
        continue
    if p.returncode:
        raise subprocess.CalledProcessError(p.returncode, (_EDITOR, _f), p.stdout, p.stderr)
    return p.returncode


def _open(_f:str):
    if not os.path.exists(_f):
        raise FileNotFoundError(_f)
    p = subprocess.Popen((_READER, _f))
    # Sleep to give a bit of time to reader to spawn
    sleep(1)
    if p.returncode:
        raise subprocess.CalledProcessError(p.returncode, (_READER, _f), p.stdout, p.stderr)
    return p.returncode


def _strip(string):
    m = re.search(_STRIP_RE, string.lower())
    return m[0] if m is not None else string


def gen_indents(indent=0, skip_first=False):
    if not skip_first:
        yield ""
    while True:
        yield " " * indent


def get_terminal_width():
    return shutil.get_terminal_size((80, 20)).columns


def check_excerpts(paper_id, paper, obj, prefix="", indents=None):
    if indents is None:
        indents = gen_indents()
    validated = True
    try:
        obj.excerpt
        _indents = gen_indents(2)
        for k, v in obj:
            if k == "value":
                _ = v
            else:
                _ = f"{k}: {v}"
            print(next(indents) + next(_indents), _, sep="")
        excerpt = _strip(obj.excerpt)
        if excerpt not in paper:
            print(next(indents) + next(_indents), f"WARNING: Could not find the exerpt [{excerpt}] justifing the value [{obj.value}] in the paper {paper_id}", sep="")
            validated = False
        return validated
    except AttributeError:
        pass

    if isinstance(obj, str):
        print(next(indents), prefix, obj, sep="")
        return True
    print(next(indents), "[", sep="")
    try:
        prefix = "* "
        _indents = gen_indents(2, skip_first=True)
        padding = 0
        for k, _ in obj:
            padding = max(padding, len(k) + 1)
        for k, v in obj:
            pad = " " * (padding - len(k))
            print(next(indents) + next(_indents), prefix, f"{k}{pad}: ", sep="", end="")
            sub_indents = gen_indents(len(next(indents)) + 2)
            _validated = check_excerpts(paper_id, paper, v, prefix="", indents=sub_indents)
            validated = validated and _validated
    except ValueError:
        try:
            prefix = "- "
            for v in obj:
                sub_indents = gen_indents(len(next(indents)) + 2, skip_first=True)
                _validated = check_excerpts(paper_id, paper, v, prefix=prefix, indents=sub_indents)
                validated = validated and _validated
        except ValueError:
            pass
    print(next(indents), "]", sep="")
    return validated


def _merge_paper_extractions(paper_id, paper, extractions: PaperExtractions, other_extractions: PaperExtractions):
    for (_, v1), (_, v2) in zip(extractions, other_extractions):
        try:
            v2.excerpt
            if not check_excerpts(paper_id, paper, v1) and check_excerpts(paper_id, paper, v2):
                v1.value = v2.value
                v1.excerpt = v2.excerpt
        except AttributeError:
            pass
    for m in other_extractions.models:
        if m not in extractions.models:
            extractions.models.append(m)
    for d in other_extractions.datasets:
        if d not in extractions.datasets:
            extractions.datasets.append(d)
    for f in other_extractions.frameworks:
        if f not in extractions.frameworks:
            extractions.frameworks.append(f)


def _iter(iterable:Iterable):
    lazy_e = None

    try:
        for k, v in iterable.items():
            yield k, v
        return
    except AttributeError as e:
        lazy_e = e

    try:
        for i, v in enumerate(iterable):
            yield i, v
        return
    except ValueError as e:
        lazy_e = e

    raise lazy_e


def _remove_duplicates(l:list):
    if l:
        last = l[0]
        yield last
    for value in l[1:]:
        if value != last:
            yield value
            last = value


def _model_dump(paper_id, paper, model:BaseModel):
    _WARNING = f"WARNING: Could not find the quote in the paper {paper_id}"

    model_dump_json = model.model_dump_json(indent=2)
    model_dump_yaml = yaml.safe_dump(
        json.loads(model_dump_json),
        sort_keys=False,
        width=120
    )

    lines = model_dump_yaml.splitlines()
    for i, l in enumerate(lines):
        if l.lstrip().startswith('quote:'):
            lstrip = l[:len(l) - len(l.lstrip())]
            end = i+1
            while end < len(lines):
                len_end_lstrip = len(lines[end]) - len(lines[end].lstrip())
                if lines[end] and len_end_lstrip <= len(lstrip):
                    break
                end += 1
            try:
                quote = yaml.safe_load("\n".join(lines[i:end]))["quote"]
            except (yaml.parser.ParserError, yaml.scanner.ScannerError):
                print(model_dump_yaml)
                print("\n".join(lines[i:end]))
                raise
            if str_normalize(quote) not in paper:
                lines.insert(end, f"{lstrip}## {_WARNING}")
    model_dump_yaml = "\n".join(lines)
    return model_dump_yaml


def _input_option(question:str, options:list):
    select:str = ""
    options = [o.lower() for o in options]
    while select not in options:
        select = input(f"{question} [{','.join(options)}]? ").lower()
    return select


def _select(key:str, *options:List[str], edit=False):
    is_equal = not edit
    for i, v in enumerate(options):
        for o in options[i+1:]:
            if v != o:
                is_equal = False
    if is_equal:
        return options[0]

    short_options = [f"{i+1}" for i, _ in enumerate(options)]
    long_options = short_options[:]
    if edit:
        short_options.append("e")
        long_options.append("edit")
    separator = "=" * max(0, 0, *map(len, sum([o.splitlines() for o in options], [])))
    separator = separator[:get_terminal_width()]

    for i, option in enumerate(options):
        print()
        prefix = f"== {key} ({i+1}) "
        print(f"{prefix}{separator[len(prefix):]}")
        print(highlight(option, YamlLexer(), TerminalTrueColorFormatter()))
    select = _input_option(f"Select {' or '.join(long_options)}", short_options)
    try:
        return options[int(select) - 1]
    except ValueError:
        pass

    # select == "e"
    return edit_content(key, "\n\n".join(options))


def edit_content(filename:str, contents:List[str]):
    tmpfile = Path(_TMPDIR.name) / f"{filename}.yaml"

    with tmpfile.open("w+t") as _f:
        _f.write(contents)

    _open_editor(str(tmpfile))
    while _input_option("Are you done with the edit", ("y","n")) != "y":
        _open_editor(str(tmpfile))

    with tmpfile.open() as _f:
        return _f.read()


def merge_paper_extractions(paper_id, paper, merged_extractions:PaperExtractions, *all_extractions: List[PaperExtractions]):
    f:Path = (ROOT_DIR / "data/merged/") / paper_id
    f = f.with_suffix(".json")

    for keys_values in zip(empty_paperextractions(), merged_extractions, *all_extractions):
        # Save progression
        f.write_text(merged_extractions.model_dump_json(indent=2))

        empty_value, merged_value, *values = [v for _, v in keys_values]

        if not [v for v in values[1:] if v != values[0]]:
            # All the values are similar to the first value
            values = values[:1]

        key = keys_values[0][0]
        attribute = f"{merged_extractions.__class__.__name__}.{key}"

        merged_value = [merged_value] if merged_value != empty_value else []
        try:
            options:List[BaseModel] = sorted(sum(values, []), key=lambda _:_.name)
            options = list(_remove_duplicates(options))
            options_str = []
            for _list in (*merged_value, options):
                concat = []
                for entry in _list:
                    prefix = "- "
                    for l in _model_dump(paper_id, paper, entry).splitlines():
                        concat.append(f"{prefix}{l}")
                        prefix = "  "
                    concat.append("")
                options_str.append("\n".join(concat))

            selection = _select(attribute, *options_str, edit=True)
            while True:
                try:
                    _selection = yaml.safe_load(selection or "[]")
                    selection = [options[0].model_validate(entry) for entry in _selection]
                    break
                except yaml.scanner.ScannerError as e:
                    print(e)
                    print("There was an error parsing the yaml. Please try again")
                    selection = edit_content(attribute, selection)
                except ValueError as e:
                    print(e)
                    print(f"There was an error validating the model {type(values[0])}. Please try again")
                    selection = edit_content(attribute, selection)
            merged_extractions.__dict__[key] = selection
            continue
        except TypeError:
            pass

        try:
            options = [_model_dump(paper_id, paper, v) for v in (*merged_value, *values)]
            selection = _select(attribute, *options, edit=True)
            while True:
                try:
                    selection = values[0].model_validate(yaml.safe_load(selection))
                    break
                except yaml.scanner.ScannerError as e:
                    print(e)
                    print("There was an error parsing the yaml. Please try again")
                    selection = edit_content(attribute, selection)
                except ValueError as e:
                    print(e)
                    print(f"There was an error validating the model {type(values[0])}. Please try again")
                    selection = edit_content(attribute, selection)
            merged_extractions.__dict__[key] = selection
            continue
        except AttributeError:
            pass

        try:
            options = [v.value for v in (*merged_value, *values)]
            selection = _select(attribute, *options, edit=True)
            while True:
                try:
                    selection = type(values[0])(selection)
                    break
                except ValueError as e:
                    print(e)
                    print("There was an error parsing the value. Please try again")
                    selection = edit_content(attribute, selection)
            merged_extractions.__dict__[key] = selection
            continue
        except AttributeError:
            pass

        merged_extractions.__dict__[key] = _select(attribute, *merged_value, *values, edit=True)


def get_papers_from_file(papers: List[str]) -> List[Tuple[str, Path, ExtractionResponse]]:

    extractions_tuple = []
    for paper in papers:
        paper_id = paper.strip()
        print('Parsing', paper_id)
        paper = (ROOT_DIR / f"data/cache/arxiv/{paper_id}.txt").read_text().lower().replace("\n", " ")
        responses = list((ROOT_DIR / "data/queries/").glob(f"{paper_id}_[0-9]*.json"))
        if not responses:
            print('No responses found for', paper_id)
            print('Skipping...')
            continue
        responses = (
            ExtractionResponse.model_validate_json(_f.read_text())
            for _f in responses
        )
        for (_,paper_id),(_,_),(_,extractions),_ in responses:
            extractions_tuple.append((paper_id, str_normalize(paper), extractions))

    extractions_tuple.sort(key=lambda _:_[0])

    return extractions_tuple


def get_papers_from_folder() -> List[Tuple[str, Path, ExtractionResponse]]:
    responses = (ROOT_DIR / "data/queries/").glob("*.json")

    extractions_tuple = []
    for response_path in responses:
        print('Parsing', response_path)
        try:
            (_,paper),(_,_),(_,extractions),_ = ExtractionResponse.model_validate_json(response_path.read_text())
        except ValidationError as e:
            print(e)
            print(f'Skipping {response_path}')
            continue
        paper_id = paper
        paper = (ROOT_DIR / "data/cache/arxiv/" / paper_id).read_text().lower().replace("\n", " ")
        extractions_tuple.append((paper_id, paper, extractions))

    extractions_tuple.sort(key=lambda _:_[0])

    return extractions_tuple


def main(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", nargs='*', type=str, default=None, help="Papers to merge")
    parser.add_argument("--input", type=Path, default=None, help="List of papers to merge")
    options = parser.parse_args(argv)

    if options.input:
        with open(options.input, "r") as f:
            papers = get_papers_from_file(f.readlines())
    if options.papers:
        papers = get_papers_from_file(options.papers)
    else:
        papers = get_papers_from_folder()

    done = []
    for i, (paper_id, paper, _) in enumerate(papers):
        if [_paper_id for (_paper_id, _, _) in done if _paper_id == paper_id]:
            continue

        print('Merging', paper_id)

        f:Path = (ROOT_DIR / "data/merged/") / paper_id
        f = f.with_suffix(".json")
        f.parent.mkdir(parents=True, exist_ok=True)

        merged_extractions = empty_paperextractions()

        if f.exists():
            try:
                merged_extractions = PaperExtractions.model_validate_json(f.read_text())
            except ValidationError as e:
                print(e)
                print(f'Invalid extraction file... Consider deleting [{f}].')
                continue
            if _input_option(
                f"The paper {paper_id} has already been merged. Do you wish to "
                f"redo the merge?",
                ("y","n")
            ) == "n":
                done.append((paper_id, paper, merged_extractions))
                continue

        all_extractions = [
            _extractions
            for _paper_id, _, _extractions in papers[i:]
            if _paper_id == paper_id
        ]

        pdf:Path = (ROOT_DIR / "data/cache/arxiv/" / paper_id).with_suffix(".pdf")
        print("Opening", pdf)
        try:
            _open(str(pdf))
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            url = f"https://arxiv.org/pdf/{pdf.stem}"
            print('Downloading from', url, 'to', str(pdf))
            urllib.request.urlretrieve(url, str(pdf))
            _open(str(pdf))

        merge_paper_extractions(paper_id, paper, merged_extractions, *all_extractions)
        done.append((paper_id, paper, merged_extractions))

        f.write_text(merged_extractions.model_dump_json(indent=2))

        print('Merged paper saved to', f)


if __name__ == "__main__":
    main()
