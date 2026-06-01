# VEX AI & Robotics Development Environment Setup Guide (Windows)

Welcome to the team! This guide will walk you through installing the required global software and cloning our core repositories to get your Windows machine ready for development.

---

## Step 1: Install VSCode & Extensions

1. **Download & Install VSCode:**

    * Go to the [Official VSCode Download Page](https://code.visualstudio.com/) and download the Windows installer.
    * Run the installer and accept the defaults (making sure **"Add to PATH"** is checked).

2. **Install the VSCode Extensions:**

    * Open VSCode.
    * Click on the **Extensions** icon on the left sidebar (or press `Ctrl + Shift + X`).
    * Install the following extensions:
        * **C/C++** by Microsoft (for C++ development code navigation).
        * **Python** by Microsoft (for AI development and virtual environment support).
        * **VEX Robotics** by VEX Robotics. *Note: This extension automatically installs the necessary underlying C++ toolchain/compiler for VEX hardware.*

---

## Step 2: Install Python 3.12 (For AI/Reinforcement Learning)

Because our AI and Reinforcement Learning libraries (like PyTorch) have explicit version requirements, you must install Python 3.12.

1. **Download Python 3.12:**

* Go to the [Python Downloads Page](https://www.python.org/downloads/).
* Look for **Python 3.12.x** (do not get 3.13+ unless explicitly instructed).

2. **Install Python:**

* Run the installer.
* **CRITICAL:** Check the box at the bottom that says **"Add python.exe to PATH"** before clicking Install Now.

3. **Verify Installation:**

* Open a new PowerShell window or Command Prompt and type:

```powershell
python --version

```

* It should return `Python 3.12.x`.

---

## Step 3: Install Git & Set Up GitHub

If you don't already have a GitHub account, go to [github.com](https://github.com/) and create one. **Once created, send your username to the team lead so you can be added to the `msoe-vex` organization.**

### 1. Install Git

* Download the installer from [git-scm.com](https://git-scm.com/download/win).
* Run the installer. You can safely click "Next" on the default settings.

### 2. Configure Your Git Identity

Open PowerShell or Terminal and run the following commands with your info:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your-github-email@example.com"

```

### 3. Generate and Set Up an SSH Key

Using SSH keys allows you to securely push and pull from GitHub without typing your credentials every time.

* **Generate the key:** Open your terminal and paste the following (replace with your GitHub email):

```powershell
ssh-keygen -t ed25519 -C "your-github-email@example.com"

```

Press `Enter` to accept the default file location, and optionally enter a memorable passphrase.

* **Copy the key to your clipboard:**

```powershell
clip < ~/.ssh/id_ed25519.pub

```

* **Add it to GitHub:**

1. Go to GitHub, click your profile picture in the top-right, and select **Settings**.
2. In the left sidebar, click **SSH and GPG keys**.
3. Click **New SSH key**, give it a title (e.g., "Windows Laptop"), and paste your key into the box.
4. Click **Add SSH key**.

---

## Step 4: Cloning the Repositories

Now that your SSH key is authorized, navigate to the folder where you want to store your team projects (e.g., `cd Documents`), and clone the target repositories:

```powershell
# Clone the Reinforcement Learning Repository
git clone git@github.com:msoe-vex/VEX-AI-Reinforcement-Learning.git

# Clone the Push-Back Repository
git clone git@github.com:msoe-vex/Push-Back.git

```

---

## Step 5: Git Crash Course (Daily Workflow)

If you haven't used Git much before you can use the VS Code Source Control interface (use `Ctrl+Shift+G` to open it), or you can use the command line. Make sure you use `cd` to enter the specific project directory before running these commands. This is the baseline workflow we use every day:

| Command | What it does | When to use it |
| --- | --- | --- |
| `git pull` | Fetches the latest code from GitHub and merges it locally. | **First thing** before you start writing any new code. |
| `git status` | Shows you what files you have changed or created. | Regular sanity check to see your progress. |
| `git add .` | Stages all your changes, preparing them to be saved. | When you've finished a feature or a chunk of work. |
| `git commit -m "msg"` | Saves your staged changes locally with a descriptive message. | Right after running `git add`. |
| `git push` | Sends your saved commits up to GitHub for the team to see. | When you're ready to share or back up your work. |

### The Standard Development Loop

```powershell
cd VEX-AI-Reinforcement-Learning   # Navigate to your active project folder
git pull                           # Always get the latest updates first
# ... write code in VSCode ...
git status                         # Review your local modifications
git add .                          # Stage your files
git commit -m "Updated local tracking nodes"   # Commit changes locally
git push                           # Push back up to the organization

```

---

## Next Steps: Repo-Specific Setup

Now that your core environment is live, open the cloned project folders in VSCode. Each repository contains a unique `README.md` file detailing exactly how to spin up its localized Python virtual environments (`.venv`), configure specific package dependencies (`requirements.txt`), and build/test the code.
