# How to contribute to tsai: pull requests made easy

## Making your first pull request to tsai

First, thank you very much for wanting to help!

To contribute to tsai, you will need to create a pull request, also known as a PR. When you submit it he tsai team get a notification and, after checking if the changes look good, we'll "merge" it (meaning your changes will be automatically added to the repository).

It's very easy to create a PR to tsai.

**Key points:**

- In this guide, we assume that you are using Linux or Mac and that you already have Anaconda or Miniconda set up. On Windows, use Ubuntu on WSL.

- We will be using GitHub's official CLI gh, which makes things faster and easier than doing things through the website.

- tsai has been built with nbdev. nbdev is a library that allows you to develop a Python library in Jupyter Notebooks, putting all your code, testing, and documentation in one place.

- We will also show you how to create a separate conda environment with all required packages. This is optional, although highly recommended.

- If you are unfamiliar with the fastai / tsai code style, be sure to read [this] (<https://docs.fast.ai/dev/style.html>) first. (Note that, like fastai, we do not follow PEP8, instead we follow a coding style designed specifically for numeric and interactive programming.)

## One-time only setup

All steps in this section need to be done **<u>only the first time you set up the environment where you will develop tsai</u>**.

**Steps:**

1. **Set up gh**. If you don't have gh (GitHub CLI) yet, you can install it by following these [instructions](https://cli.github.com/manual/installation). To authenticate with your GitHub account run:

   ```
   gh auth login
   ```

   and follow the instructions.
2. **Create and activate a new conda environment**. This step is optional, but recommended. You will need to choose a name for the environment and a version of Python (i.e. `tsai_dev` and Python 3.9).

   ```
   conda create -n tsai_dev python=3.9
   conda activate tsai_dev
   ```

3. **Set up tsai**. We'll use an editable install.
   Navigate to the folder where you want to install the tsai repo in your local machine and:

   ```
   git clone https://github.com/timeseriesAI/tsai
   cd tsai
   pip install -e "tsai[dev]"
   ```

   The last step will install all packages required to run tsai.

4. **Set up git hooks**. This step is required by nbdev. Run:

   ```
   nbdev_install_hooks
   ```

   in the tsai repo folder. Git hooks clean up the notebooks to remove the extraneous stuff stored in the notebooks. In this way you avoid unnecessary merge conflicts.

Great! You are now ready to start working on your first tsai PR.

## Creating a PR

If you have already set up your environment, you can proceed with the following steps.

**Steps:**

1. Navigate to the tsai repo folder and activate your tsai_dev environment:

   ```
   conda activate tsai_dev
   ```

2. Use

   ```python
   git pull origin main
   ```

   to ensure you have the most recent version of the repo.

3. Navigate to the nbs folder in the local repo and open the required Jupyter Notebook you wish to modify. If you are unsure which notebook it is, you can find it at the top of the script you wish to modify. Make whatever changes you want to make in the notebooks (ie., update feature, add new feature, add tests, add documentation, etc.).

   - Any new code you create needs to be in a cell with ```#|export``` at the top.
   - You can write Markdown text as you would normally do to document any new functionality.
   - Remember that nbdev creates all scripts, tests and documentation from the notebooks. That's why it's important that you explain how the code works. You should also add tests. Tests will ensure that the new functionality keeps working in the future, when new changes are made.
   - When finished, restart the kernel and re-run the whole notebook. Make sure everything runs smoothly until the end. This will also automatically save the notebook. If you see this at the bottom of the notebook "Correct conversion! 😃", it means everything went well and you can close the notebook. You are now ready to create your PR.

4. In your terminal, run

   ```python
   nbdev_prepare
   ````

    This will export, test, and clean notebooks, and render README if needed.

7. Review the files that have been changed. You can do that with:

   ```
   git status
   ```

   Whenever you change a notebook, you should see at least see changes in its .ipynb and .py corresponding files.

4. Add your code and commit your changes:

   ```pyhon
   git commit -am "add a message here"
   ```

6. You are now ready to create a PR:

   ```python
   gh pr create --title "brief PR title"
   ```

   You'll get some questions:
   - ? Where should we push the 'main' branch? Select: `Create a fork of timeseriesAI/tsai`
   - ? Body: You can enter a brief description of the PR, do it later in GitHub or press Enter and skip this step.
   - ? What's next? Select `Submit` and Enter. That will create the PR.

8. Optional: click on the link that will show up. That'll take you to the PR in GitHub. You can now edit the description to provide details on the PR, and click "Update comment". That's it.

You'll soon receive a response from a tsai team member.
Once your PR passess all tests and is merged you'll become a `tsai` contributor 🚀.
