# Environment Setup for DROID Deployment (uv)

This document describes how to set up **DROID deployment using a uv-managed environment**.

---

## 0. Environment Setup (uv)

```bash
cd $RLDX_ROOT

uv venv --python 3.10
uv sync
```

Optional:

```bash
source .venv/bin/activate
```

---

## 1. Python Dependencies

Install the required Python packages:

```bash
uv add moviepy==1.0.3
uv pip install gym==0.26.2
uv pip install zerorpc==0.6.3
```

Install PyTorch for RTX 5090 / Blackwell support:

```bash
uv pip install \
  torch==2.8.0+cu129 \
  torchvision==0.23.0+cu129 \
  --index-url https://download.pytorch.org/whl/cu129
```

Install torchaudio:

```bash
uv pip install \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

> Note: after installation, verify that your environment is actually using the intended torch build.

```bash
./.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Expected output:

```text
2.8.0+cu129 12.9
```

---

## 2. OpenCV Compatibility Fix for DROID

DROID relies on older ArUco / Charuco OpenCV APIs such as:

- `aruco.Dictionary_get`
- `aruco.CharucoBoard_create`

Remove existing OpenCV packages first:

```bash
uv pip uninstall \
  opencv-python \
  opencv-python-headless \
  opencv-contrib-python \
  opencv-contrib-python-headless
```

Install the compatible version:

```bash
uv pip install \
  opencv-contrib-python==4.6.0.66 \
  opencv-python==4.6.0.66
```

Verify:

```bash
./.venv/bin/python -c "import cv2; from cv2 import aruco; print(cv2.__version__); print(hasattr(aruco, 'Dictionary_get')); print(hasattr(aruco, 'CharucoBoard_create'))"
```

Expected output:

```text
4.6.0
True
True
```

---

## 3. Flash-Attention Fix for Qwen3-VL / RLDX

If you encounter an error like:

```text
ImportError: ... flash_attn_2_cuda ... undefined symbol ...
```

it usually means the installed `flash-attn` binary is incompatible with the current torch version.

### 3.1 Remove old flash-attn

```bash
cd $RLDX_ROOT
uv pip uninstall flash-attn
```

### 3.2 Install flash-attn v2.8.3 from source against the current uv environment

```bash
git clone https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention
cd /tmp/flash-attention
git checkout v2.8.3
uv pip install --python $RLDX_ROOT/.venv/bin/python --no-build-isolation .
```

Verify:

```bash
$RLDX_ROOT/.venv/bin/python -c "import torch, flash_attn; print(torch.__version__, torch.version.cuda); print(flash_attn.__file__)"
```

Expected output should show:

- `torch==2.8.0+cu129`
- a valid `flash_attn` import path inside the uv environment

Example:

```text
2.8.0+cu129 12.9
$RLDX_ROOT/.venv/lib/python3.10/site-packages/flash_attn/__init__.py
```

---

## 4. TorchCodec and FFmpeg

Install the compatible TorchCodec version:

```bash
cd $RLDX_ROOT
uv pip install torchcodec==0.7.0
```

Install ffmpeg:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Verify ffmpeg:

```bash
ffmpeg -version
```

---

## 5. ZED SDK and `pyzed==5.0` Installation

### 5.1 Download ZED SDK

```bash
curl -L -A "Mozilla/5.0" -o zedsdk_5.0_amd64.run \
  https://download.stereolabs.com/zedsdk/5.0/cu12/ubuntu24
```

### 5.2 Install required system package

```bash
sudo apt update
sudo apt install -y zstd
```

### 5.3 Install ZED SDK

```bash
chmod +x zedsdk_5.0_amd64.run
sudo ./zedsdk_5.0_amd64.run
```

### 5.4 Install the `pyzed` wheel into the uv environment

```bash
sudo cp /usr/local/zed/pyzed-5.0-cp310-cp310-linux_x86_64.whl ~/
sudo chown "$USER":"$USER" ~/pyzed-5.0-cp310-cp310-linux_x86_64.whl

uv pip install ~/pyzed-5.0-cp310-cp310-linux_x86_64.whl
```

---

## 6. Install DROID

Install DROID:

```bash
uv pip install git+https://github.com/droid-dataset/droid.git
```

Then install it in editable mode if needed:

```bash
cd droid
uv pip install -e .
```

---

## 7. DROID Configuration

### 7.1 Update `droid/droid/misc/parameters.py`

Replace the file contents with:

```python
from cv2 import aruco

# Robot Params #
nuc_ip = "172.30.1.112"
robot_ip = "172.16.0.2"
laptop_ip = "172.30.1.103"
sudo_password = "rlwrld!@#$"
robot_type = "fr3"  # 'panda' or 'fr3'
robot_serial_number = "295341-0051724"

# Camera ID's #
hand_camera_id = "10623639"
varied_camera_1_id = "34022131"
varied_camera_2_id = "34022131"

# Charuco Board Params #
CHARUCOBOARD_ROWCOUNT = 9
CHARUCOBOARD_COLCOUNT = 14
CHARUCOBOARD_CHECKER_SIZE = 0.020
CHARUCOBOARD_MARKER_SIZE = 0.016
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_100)

# Ubuntu Pro Token (RT PATCH) #
ubuntu_pro_token = "1LrMq6bwozpTZQydy1Dx2iUHK2tz"

# Code Version [DONT CHANGE] #
droid_version = "1.3"
```

### 7.2 Modify `droid/droid/misc/server_interface.py`

Change:

```python
func_list = [self.launch_controller, self.launch_robot]
```

to:

```python
func_list = [self.launch_robot]
```

---

## 8. User Permission Setup

Ensure your user belongs to the required groups:

- `sudo`
- `zed`
- `video`

Run:

```bash
sudo usermod -aG sudo zed video "$USER"
sudo reboot
```

---

## 9. Running

Use the uv-managed Python explicitly:

```bash
$RLDX_ROOT/.venv/bin/python \
  run_scripts/deploy/droid_deploy.py ...
```

or in a shell script:

```bash
"$BASE_DIR/.venv/bin/python" -u ...
```

---

## 10. Quick Sanity Checks

### Torch

```bash
./.venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### Flash-Attention

```bash
./.venv/bin/python -c "import flash_attn; print(flash_attn.__file__)"
```

### OpenCV ArUco compatibility

```bash
./.venv/bin/python -c "import cv2; from cv2 import aruco; print(cv2.__version__); print(hasattr(aruco, 'Dictionary_get')); print(hasattr(aruco, 'CharucoBoard_create'))"
```

### Zerorpc

```bash
./.venv/bin/python -c "import zerorpc; print(zerorpc.__version__)"
```

### Gym

```bash
./.venv/bin/python -c "import gym; print(gym.__version__)"
```