#

We welcome any kind of contribution to our software, from simple comment or
question to a full fledged [pull
request](https://help.github.com/articles/about-pull-requests/). Please read and
follow our [Code of Conduct](CODE_OF_CONDUCT.md).

A contribution can be one of the following cases:

1. you have a question;
1. you think you may have found a bug (including unexpected behavior);
1. you want to make some kind of change to the code base (e.g. to fix a bug, to add a new feature, to update documentation);
1. you want to make a new release of the code base.

The sections below outline the steps in each case.

## You have a question

1. use the search functionality [in
   issues](https://github.com/hybridlabs-nl/fowt_ml/issues) to see if someone
   already filed the same issue;
2. if your issue search did not yield any relevant results, make a new issue;
3. apply the "Question" label; apply other labels when relevant.

## You think you may have found a bug

1. use the search functionality [in
   issues](https://github.com/hybridlabs-nl/fowt_ml/issues) to see if someone
   already filed the same issue;
2. if your issue search did not yield any relevant results, make a new issue, making sure to provide enough information to the rest of the community to understand the cause and context of the problem. Depending on the issue, you may want to include:
    - the [SHA hashcode](https://help.github.com/articles/autolinked-references-and-urls/#commit-shas) of the commit that is causing your problem;
    - some identifying information (name and version number) for dependencies you're using;
    - information about the operating system;
3. apply relevant labels to the newly created issue.

## You want to make some kind of change to the code base

1. (**important**) announce your plan to the rest of the community *before you start working*. This announcement should be in the form of a (new) issue;
2. (**important**) wait until some kind of consensus is reached about your idea being a good idea;
3. follow the instruction in [developer_guide.md](developer_guide.md).

In case you feel like you've made a valuable contribution, but you don't know
how to write or run tests for it, or how to generate the documentation: don't
let this discourage you from making the pull request; we can help you! Just go
ahead and submit the pull request, but keep in mind that you might be asked to
append additional commits to your pull request.

## You want to make a new release of the code base

To create a release you need write permission on the repository.

1. Check the author list in [`CITATION.cff`](../CITATION.cff)
2. Bump the version using `bump-my-version bump <major|minor|patch>`. The
   version can be manually changed in `pyproject.toml` in
   the root of the repository. Follow [Semantic Versioning](https://semver.org/)
   principles. Also, update `__version__` variable in `src/fwot_ml/__init__.py`
   to the same version.
3. Go to the [GitHub release
   page](https://github.com/hybridlabs-nl/FOWT-ML/releases). Press draft a new
   release button. Fill version, title and description field. Press the Publish
   Release button. For this package, if the zenodo integration is enabled, a new
   DOI will be created automatically.
4. This software automatically publish to PyPI using a release or publish
   workflow. Wait until [PyPi publish
   workflow](../.github/workflows/python-publish.yml) has completed and verify
   new release is on [PyPi](https://pypi.org/project/matchms/#history).
