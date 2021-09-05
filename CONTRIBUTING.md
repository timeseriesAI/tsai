# How to contribute to tsai: pull requests made easy

> Making your first pull request to fastai

First, thank you very much for wanting to help!

If you're planning to contribute code that is unrelated to an existing issue, it's a good idea to open a new issue describing your proposal before starting work on it. The project maintainers might give you feedback that will help to shape your work, which will ultimately increase the likelihood that your pull request will be accepted.

To contribute to tsai, you will need to create a pull request, also known as a PR. We'll get a notification when a pull request comes in, and after checking if the changes look good we'll "merge" it (meaning we click a button on GitHub that causes all of those changes to be automatically added to the repository).

Making a pull request for the first time can be a bit overwhelming, so we've put together this guide to help you get started.


**Key points:**

- In this guide, we assume that you are using Linux or Mac and that you already have Anaconda or Miniconda set up. On Windows, use Ubuntu on WSL.

- We will be using GitHub's official command line tool gh, which makes things faster and easier than doing things through the website.

- tsai has been built with nbdev. nbdev is a library that allows you to develop a Python library in Jupyter Notebooks, putting all your code, testing, and documentation in one place.

- We will also show you how to create a separate conda environment with all required packages. This is optional, although highly recommended.

- If you are unfamiliar with the fastai / tsai code style, be sure to read [this] (https://docs.fast.ai/dev/style.html) first. (Note that, like fastai, we do not follow PEP8, instead we follow a coding style designed specifically for numeric and interactive programming.)

## One-time only setup:

All steps in this section need to be done **<u>only the first time you set up the environment where you will develop tsai</u>**.

**Steps:**

1. **Set up gh**. If you don't have gh (GitHub CLI) yet, you can install it by following these [instructions](https://cli.github.com/manual/installation). To authenticate with your GitHub account run:
   ```
   gh auth login
   ```
   and follow the instructions.
2. **Create and activate a new conda environment**. You will need to choose a name for the environment and a version of Python (I chose `tsai_dev` and Python 3.8, but you can choose others). 
   ```
   conda create -n tsai_dev python=3.8
   conda activate tsai_dev
   ```
3. **Set up tsai**. We'll use an editable install. 
   Navigate to the folder where you want to install the tsai repo in your local machine and:
   ```
   gh repo clone https://github.com/timeseriesAI/tsai
   cd tsai
   pip install -e ".[dev]"
   ```  
   The last step will take 4-5 minutes and will install all packages required to run tsai.
   
4. **Set up git hooks**. This step is required by nbdev. Run:
   ```
   nbdev_install_git_hooks
   ``` 
   inside the same tsai repo folder. Git hooks clean up the notebooks to remove the extraneous stuff stored in the notebooks. In this way you avoid unnecessary merge conflicts.
   
Great! You are now ready to start working on your first tsai PR. 

## Creating your PR:

If you have already set up your environment, you can proceed with the following steps.

**Steps:**

1. Activate your conda environment. We'll assume the name of the environment you created is tsai_env (if you used a different name just replace it by that name). Navigate to the tsai repo folder and run: 

   ```
   conda activate tsai_dev
   ```

2. Create a new git branch. This is where new feature/bug-fix should be made. Replace my-branch-name by something that is descriptive of the change and will be easy for you to remember in the future if you need to update your PR. Before you start making any changes to your local files, it's a good practice to first synchronize your local repository with the project repository. 
   
   ```
   git pull
   git checkout -b my-branch-name
   ``` 
   If you just need to navigate to a previously created branch just run: 
   ```
   git checkout my-branch-name
   ```
3. Navigate to the nbs folder and open the required Jupyter Notebook you wish to modify. If you are unsure which notebook it is, you can find it at the top of the script you wish to modify. Make whatever changes you want to make in the notebooks (ie., update feature, add new feature, add tests, add documentation, etc.). 

   - Any new code you create needs to be in a cell with ```#export``` at the top. 
   - You can write Markdown text as you would normally do to document any new functionality. 
   - Remember that nbdev creates all scripts, tests and documentation from the notebooks. That's why it's important that you explain how the code works. You should also add tests. Tests will ensure that the new functionality keeps working in the future, when new changes are made.
   - When finished, restart the kernel and re-run the whole notebook. Make sure everything runs smoothly until the end. This will also automatically save the notebook. If you see this at the bottom of the notebook "Correct conversion! 😃", it means everything went well and you can close the notebook. 
   
   Note: the create_scripts() function at the bottom of all nbs will automatically save the notebook and run nbdev_build_lib which converts the notebooks to scripts. There's no need to do that manually in tsai. 
    
4. Check that the local library and all notebooks match. The script:

   ```
   nbdev_diff_nbs
   ```

   can let you know if there is a difference between the local library and the notebooks. If everything's ok, you shouldn't get anything back when running the script.
    
5. Run all library tests using

   ```
   nbdev_test_nbs
   ```
 
   in your terminal. This will run all the tests in tsai and will take a few minutes. 
    
   Sometimes, a change you made creates an issue in the same or in other notebook/s. In that case, you'll get a message like this: 
    
   Exception: The following notebooks failed:
   003_data.preprocessing.ipynb
    
   You should then open that notebook and run it to learn where the issue is. You should fix it, re-run the notebook you have fixed, close it and re-run:

   ```
   nbdev_test_nbs
   ```
    
   again until you get this message: All tests are passing!    
    
6. Update the documentation by running: 
    
   ```
   nbdev_build_docs
   ```   
   
7. Review the files that have been changed. You can do that with:

   ```
   git status
   ```
   
   Whenever you make a change in a notebook, you should see at least see changes in its .ipynb, .py and .html corresponding files.

8. Commit changes with a brief message describing your changes. For example:

   ```
   git commit -am "added training chart to index nb"
   ```
   
9. Create a pull request using GitHub CLI. 
    
   ```
   gh pr create -B main -t "enter title" -b "enter body of PR here"
   ```
    
   You can link an issue to the pull request by referencing the issue in the body of the pull request. If the body text mentions Fixes #123 or Closes #123, the referenced issue will automatically get closed when the pull request gets merged.
    
   This command will automatically create a fork for you if you’re in a repository that you don’t have permission to push to.
   
And that's it! If you navigate to https://github.com/timeseriesAI/tsai/pulls you should see your PR there.

10. Updating a PR:
   If you need to change your code after a PR has been created you can do it by sending more commits to the same remote branch. For example:
   ```
   git checkout -b my-branch-name
   ```
   Repeat steps 3 through 7 
   ```
   git commit -m "add a relevant commit message"
   git push
   ```
   Your new commit will automatically show up in the PR on the github page. If these are small changes they can be squashed together at the merge time and appear as a single commit in the repository.
   
10. To return to the main branch use: 
   ```git checkout main```
   To return to the base environment use:
   ```conda deactivate```

## Post-PR steps:

In the future, once your PR has been merged or rejected, you can delete your branch if you don't need it any more.

1. Make sure you are in the main branch: 
   ```
   git checkout main
   ```
   
2. Delete the branch you no longer need by running either of these 2 commands:
   ```
   git branch -d my-branch-name
   ```
   or 
   ```
   git branch -D my-branch-name
   ```
   The -d option only deletes the branch if it has already been merged. The -D option is a shortcut for --delete --force, which deletes the branch irrespective of its merged status.
   
3. If you no longer plan to contribute to tsai and want to delete / remove the environment, type the following in your terminal:
   ```
   conda env remove --name tsai_dev
   ```  
