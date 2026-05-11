# turtlebot_rl

Reinforcement learning project targeting the **TurtleBot3 Burger**. Train in
Gazebo, then deploy to real hardware. End goals: autonomous exploration /
mapping and object following.

The full phased plan lives in `~/.claude/plans/i-wanna-start-a-streamed-lemon.md`.
This README is the operator's manual.

## Hardware target

- TurtleBot3 Burger (lidar + IMU; we add a virtual camera in sim for Phase 3).
- Training rig: Windows desktop with RTX 4070 Ti, running Ubuntu 22.04 in WSL2.

## One-time setup (Windows host)

You'll do all the Linux work inside WSL2. CUDA passes through to the 4070 Ti,
and Gazebo's GUI renders on Windows via WSLg.

### 1. Install WSL2 + Ubuntu 22.04

Run in PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu-22.04
```

Reboot. Launch Ubuntu from the Start menu, set a username/password.

Verify WSLg works:

```bash
sudo apt update && sudo apt install -y x11-apps
xeyes   # a little GUI window should appear on your Windows desktop
```

### 2. Install the NVIDIA CUDA driver for WSL2

On **Windows** (not Ubuntu), install the latest NVIDIA Game Ready or Studio
driver. That driver ships the WSL2 GPU stub — you do not install a separate
Linux NVIDIA driver inside WSL2.

Inside Ubuntu, verify:

```bash
nvidia-smi    # should list the 4070 Ti
```

Then install the CUDA toolkit (for PyTorch builds that need it):

```bash
# Follow https://developer.nvidia.com/cuda-downloads — pick:
#   Linux > x86_64 > WSL-Ubuntu > 2.0 > deb (network)
```

### 3. Install ROS 2 Humble

Inside Ubuntu 22.04:

```bash
sudo apt install -y software-properties-common curl
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
ros2 run demo_nodes_cpp talker   # should print "Publishing: 'Hello World: 1'"
```

### 4. Install TurtleBot3 packages + Gazebo

```bash
sudo apt install -y \
  ros-humble-gazebo-* \
  ros-humble-turtlebot3 \
  ros-humble-turtlebot3-simulations \
  ros-humble-turtlebot3-msgs \
  ros-humble-cartographer ros-humble-cartographer-ros \
  ros-humble-nav2-bringup
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
source ~/.bashrc
```

Verify the simulator launches:

```bash
ros2 launch turtlebot3_gazebo empty_world.launch.py
```

You should see Gazebo open on your Windows desktop with the Burger spawned.

### 5. Python environment for RL

```bash
sudo apt install -y python3-venv python3-pip
cd ~/turtlebot_rl   # or wherever you cloned this repo
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Verify PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA GeForce RTX 4070 Ti
```

## Per-shell setup

Every new shell needs ROS + venv sourced:

```bash
source /opt/ros/humble/setup.bash
source ~/turtlebot_rl/ros2_ws/install/setup.bash  # once we've built the workspace
source ~/turtlebot_rl/.venv/bin/activate
export TURTLEBOT3_MODEL=burger
```

Stick those in `~/.bashrc` once everything's stable.

## Repo layout

```
turtlebot_rl/
├── envs/
│   ├── goal_nav_env.py        # Phase 1 — reach a target pose
│   ├── exploration_env.py     # Phase 2 — coverage / mapping
│   └── follow_env.py          # Phase 3 — object following
├── train.py                   # Stable-Baselines3 entry point
├── eval.py                    # roll out a saved policy
└── ros_bridge.py              # /cmd_vel, /scan, /odom helpers

ros2_ws/src/turtlebot3_rl_bringup/
├── launch/                    # launch files that bring up Gazebo + RL node
└── worlds/                    # custom Gazebo worlds (training arenas)

runs/                          # TensorBoard logs + checkpoints (gitignored)
```

## Phase 1 quick start (once setup is done)

```bash
# Terminal 1 — simulator
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2 — training
source .venv/bin/activate
python -m turtlebot_rl.train --env goal_nav --algo ppo --total-steps 1_000_000

# Terminal 3 — watch learning
tensorboard --logdir runs/
```

## Status

- [x] Phase 0 — repo scaffolded
- [ ] Phase 0 — WSL2 + ROS 2 Humble + Gazebo installed and verified
- [ ] Phase 1 — goal-reaching navigation policy trained
- [ ] Phase 2 — exploration policy
- [ ] Phase 3 — object following with camera
- [ ] Phase 4 — sim-to-real on hardware Burger
- [ ] Phase 5 — onboard chat
