# Change Detection based on MMSegmentation

## Description

Author： @KKIEEK

This project implements a baseline for the bi-temporal change detection task, which is simply concatenates intermediate features.

## Usage

### Prerequisites

- Python 3.7
- PyTorch 1.6 or higher
- [MIM](https://github.com/open-mmlab/mim) v0.33 or higher
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) v1.0.0rc2 or higher

### Preparing dataset

You can download dataset for change detection in [this link](https://justchenhao.github.io/LEVIR).

```shell
tree data

data
└── levir
    ├── train
    │   ├── A
    │   │   ├── xxx{img_suffix}
    │   │   ├── yyy{img_suffix}
    │   │   └── zzz{img_suffix}
    │   ├── B
    │   │   ├── xxx{img_suffix}
    │   │   ├── yyy{img_suffix}
    │   │   └── zzz{img_suffix}
    │   ├── label
    │   │   ├── xxx{seg_map_suffix}
    │   │   ├── yyy{seg_map_suffix}
    │   │   └── zzz{seg_map_suffix}
    ├── val
```

### Training commands

```shell
mim train mmsegmentation configs/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-20k_levir-512x512.py
```

### Testing commands

```shell
mim test mmsegmentation configs/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-20k_levir-512x512.py --checkpoint $CHECKPOINT_PATH
```

### LEVIR-CD

| Method  | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | mIoU  | mIoU(flip) | config                                                                                | download                 |
| ------- | -------- | --------- | ------- | -------- | -------------- | ----- | ---------- | ------------------------------------------------------------------------------------- | ------------------------ |
| UPerNet | Swin-T   | 512x512   | 20000   | 18.915   |                | 85.18 | 85.33      | [config](configs/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-20k_levir-512x512.py) | [model](<>) \| [log](<>) |

## Checklist

Here is a checklist illustrating a usual development workflow of a successful project, and also serves as an overview of this project's progress.

> The PIC (person in charge) or contributors of this project should check all the items that they believe have been finished, which will further be verified by codebase maintainers via a PR.

> OpenMMLab's maintainer will review the code to ensure the project's quality. Reaching the first milestone means that this project suffices the minimum requirement of being merged into 'projects/'. But this project is only eligible to become a part of the core package upon attaining the last milestone.

> Note that keeping this section up-to-date is crucial not only for this project's developers but the entire community, since there might be some other contributors joining this project and deciding their starting point from this list. It also helps maintainers accurately estimate time and effort on further code polishing, if needed.

> A project does not necessarily have to be finished in a single PR, but it's essential for the project to at least reach the first milestone in its very first PR.

- [ ] Milestone 1: PR-ready, and acceptable to be one of the `projects/`.

  - [ ] Finish the code

> The code's design shall follow existing interfaces and convention. For example, each model component should be registered into `mmseg.registry.MODELS` and configurable via a config file.

- [ ] Basic docstrings & proper citation

> Each major object should contain a docstring, describing its functionality and arguments. If you have adapted the code from other open-source projects, don't forget to cite the source project in docstring and make sure your behavior is not against its license. Typically, we do not accept any code snippet under GPL license. [A Short Guide to Open Source Licenses](https://medium.com/nationwide-technology/a-short-guide-to-open-source-licenses-cf5b1c329edd)

- [ ] Test-time correctness

> If you are reproducing the result from a paper, make sure your model's inference-time performance matches that in the original paper. The weights usually could be obtained by simply renaming the keys in the official pre-trained weights. This test could be skipped though, if you are able to prove the training-time correctness and check the second milestone.

- [ ] A full README

> As this template does.

- [ ] Milestone 2: Indicates a successful model implementation.

  - [ ] Training-time correctness

> If you are reproducing the result from a paper, checking this item means that you should have trained your model from scratch based on the original paper's specification and verified that the final result matches the report within a minor error range.

- [ ] Milestone 3: Good to be a part of our core package!

  - [ ] Type hints and docstrings

> Ideally *all* the methods should have [type hints](https://www.pythontutorial.net/python-basics/python-type-hints/) and [docstrings](https://google.github.io/styleguide/pyguide.html#381-docstrings). [Example](https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/utils/io.py#L9)

- [ ] Unit tests

> Unit tests for each module are required. [Example](https://github.com/open-mmlab/mmsegmentation/blob/main/tests/test_utils/test_io.py#L14)

- [ ] Code polishing

> Refactor your code according to reviewer's comment.

- [ ] Metafile.yml

> It will be parsed by MIM and Inferencer. [Example](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/fcn/fcn.yml)

- [ ] Move your modules into the core package following the codebase's file hierarchy structure.

> In particular, you may have to refactor this README into a standard one. [Example](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/fcn/README.md)

- [ ] Refactor your modules into the core package following the codebase's file hierarchy structure.
