# VEX AI and Robotics Development Environment Setup Guide

This guide is the general setup checklist for new contributors. Follow it in order so your machine is ready for cloning repositories, editing code in VS Code, and using Git safely.

## Step 1: Install VS Code

1. Download VS Code from [code.visualstudio.com](https://code.visualstudio.com/).
2. Run the installer and accept the default options.
3. Make sure the option to add VS Code to your PATH is enabled if the installer offers it.
4. Launch VS Code once the install is complete.

## Step 2: Install the recommended VS Code extensions

Open the Extensions view in VS Code with Ctrl+Shift+X and install these extensions:

1. [**Python** by Microsoft](https://marketplace.visualstudio.com/items?itemName=ms-python.python).
2. [**C/C++** by Microsoft](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools).
3. [**VEX Robotics** by VEX Robotics](https://marketplace.visualstudio.com/items?itemName=VEXRobotics.vexcode).
4. [**Remote - SSH** by Microsoft](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh).
5. [**WSL** by Microsoft](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl).

The Python extension gives you interpreter selection and virtual environment support. The C/C++ extension improves navigation and IntelliSense. The VEX Robotics extension provides VEX-specific tooling.

## Step 3: Install Python 3.12

This project uses Python 3.12. Install that version unless a repository README says something different.

1. Download Python 3.12.x from [python.org](https://www.python.org/downloads/).
2. Run the installer.
3. Check the box that says Add python.exe to PATH before you finish the install.
4. Open a new PowerShell window or Command Prompt.
5. Verify the install with:

```powershell
python --version
```

You should see Python 3.12.x. If you do not, stop and fix the PATH or installation before continuing.

## Step 4: Install Git

1. Download Git from [git-scm.com](https://git-scm.com/download/win).
2. Run the installer and accept the default settings unless you have a reason not to.
3. Open a new terminal and verify Git is installed:

```powershell
git --version
```

If the command is not found, Git was not installed correctly or your terminal needs to be restarted.

## Step 5: Create your GitHub account and set your identity

If you do not already have a GitHub account, create one at [github.com](https://github.com/).

Once you have an account, set your global Git name and email. Use the same email address that is attached to your GitHub account:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your-github-email@example.com"
```

Verify the values with:

```powershell
git config --global --list
```

## Step 6: Generate and add an SSH key

SSH keys let Git authenticate to GitHub without asking for your password every time.

You may need to repeat this step if you also want to use SSH to use GitHub from WSL or ROSIE, but start with setting it up on your local machine first.

You can also reuse this same SSH key for both GitHub and ROSIE if you want, but that is optional. If you want to use the same key, add it to GitHub first, then follow the ROSIE instructions to add the same key there.

1. Open PowerShell or Command Prompt.
2. Generate a key:

```powershell
ssh-keygen -t ed25519 -C "your-github-email@example.com"
```

3. Press Enter to accept the default file location.
4. If prompted for a passphrase, choose one you can remember.
5. Copy the public key to your clipboard:

```powershell
clip < ~/.ssh/id_ed25519.pub
```

6. In GitHub, go to Settings, then SSH and GPG keys.
7. Click New SSH key.
8. Give the key a clear title, such as your laptop name.
9. Paste the key into the box and save it.

Test the connection after adding the key:

```powershell
ssh -T git@github.com
```

The first time you connect, GitHub may ask you to confirm its host key. That is normal.

## Step 7: Set up the repositories

After the general setup above, follow the setup instructions for each repository you plan to work with:

1. [msoe-vex/Push-Back](https://github.com/msoe-vex/Push-Back)
2. [msoe-vex/VEX-AI-Reinforcement-Learning](https://github.com/msoe-vex/VEX-AI-Reinforcement-Learning)
3. [msoe-vex/VAIC_25_26](https://github.com/msoe-vex/VAIC_25_26)

Each repository has its own README with the project-specific clone, environment, run, and workflow instructions.
