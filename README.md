Updated:20250418 16:00

因为本人买了10卡gpu服务器做测试，发现vllm对x99平台的多卡机器优化可能有问题，所以研究了另外一个张量并行的框架。

下面有请 mlc-llm https://github.com/mlc-ai/mlc-llm

很开心的说，安装该框架比较简单，而且该框架在amd架构的多卡机系统中有很高的gpu利用率，是目前发现的对amd的老卡支持最好的，但美中不足的是，模型需要特制模型，因此使用此框架唯一的缺点是需要重新下模型，而且只有一部分特制模型。
应该可以对现有模型转换但是我还没有尝试。
```
测试结果 mi50*8
qwq 32b q4f16 -----1卡16-18token/s，并发没测(10卡机实测)
qwq 32b q4f16 -----2卡29-36token/s，并发60token/s (36token是特殊双路epyc4代环境跑出来的)
qwq 32b q4f16 -----4卡30-35token/s,并发150-180token/s(10卡机实测)
qwq 32b q4f16 -----8卡38-41token/s,并发280-300token/s(10卡机实测)
```
```
已经安装conda的用户跳过此步骤，已经安装conda的用户跳过此步骤，已经安装conda的用户跳过此步骤
建议使用conda作为虚拟环境
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda init
conda --version
conda config --set auto_activate_base ture
conda create --name mlc python=3.10
conda activate mlc
```

```
已经安装rocm的用户跳过此步骤，已经安装rocm的用户跳过此步骤，已经安装rocm的用户跳过此步骤

AMD官方rocm安装(不建议使用官方的，因为他现在默认是6.4)
https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html

sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
wget https://repo.radeon.com/amdgpu-install/6.2.2/ubuntu/jammy/amdgpu-install_6.2.60202-1_all.deb
sudo apt install ./amdgpu-install_6.2.60202-1_all.deb
sudo apt update
sudo apt install amdgpu-dkms rocm
# linker
sudo tee --append /etc/ld.so.conf.d/rocm.conf <<EOF
/opt/rocm/lib
/opt/rocm/lib64
EOF
sudo ldconfig

# path
export PATH=$PATH:/opt/rocm-6.2.2/bin

# verify drivers
dkms status

# You need to close all windows and reboot to make changes effective.
reboot
```


下面开始安装mlc-llm
```
已经创建该conda环境的用户跳过此步骤 conda create --name mlc python=3.10
conda activate mlc
conda install -c conda-forge libstdcxx-ng
只需要这一条命令即可安装全部的环境，摆脱编译的烦恼(如果遇到timeout，可能需要开启代理)
python -m pip install --pre -U -f https://mlc.ai/wheels mlc-llm-nightly-rocm62 mlc-ai-nightly-rocm62
```

下载模型qwq 32b q4f16(其余模型请到下面链接自己找),请只下载f16_1类型的模型,f16_0这种无法多卡并行

https://huggingface.co/mlc-ai/QwQ-32B-q4f16_1-MLC/tree/main
```
温馨提示:git clone应该无需代理
安装git-lfs
git lfs install
如果上面的命令不好使试试这个
apt install git-lfs

git clone https://huggingface.co/mlc-ai/QwQ-32B-q4f16_1-MLC
下载完模型后会在本地创建QwQ-32B-q4f16_1-MLC目录
可以新开一个窗口不停使用命令查看目前下载的进度，git clone下载大文件不会显示进度
du -sh QwQ-32B-q4f16_1-MLC
```

运行推理
```
mlc_llm serve /root/QwQ-32B-q4f16_1-MLC  --mode server --overrides "tensor_parallel_shards=2" --host 0.0.0.0

温馨提示:
1.rocm-smi -b 可以查看每张pcie的实时带宽
2.q0f16是指未量化的 float16 格式
3.如果要在多个 GPU 上启用张量并行，请 向配置生成命令添加参数。--tensor-parallel-shards $NGPU
4.MLC 支持的量化完整列表 https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/quantization/quantization.py#L29
5.MLC LLM 中的引擎模式。我们提供了三种预设模式：local、interactive和server。默认模式为local。
  5.1 local是指本地服务器部署，请求并发度较低。因此，最大批处理大小将设置为 4，最大总序列长度和预填充块大小将设置为模型的上下文窗口大小（或滑动窗口大小）。
  5.2 interactive模式是指服务器以交互式方式使用，最多只能处理 1 个并发请求。因此，最大批处理大小将设置为 1，最大总序列长度和预填充块大小将设置为模型的上下文窗口大小（或滑动窗口大小）。
  5.3 server模式指的是大型服务器用例，它可能处理大量并发请求，并希望尽可能多地使用 GPU 内存。在此模式下，我们将自动推断最大可能的最大批次大小和最大总序列长度。
6. 更多的内容查看 https://llm.mlc.ai/docs/deploy/rest.html#rest-launch-server
```

triton的替换
```
因为不清楚mlc所安装的是否是魔改triton，如果你对性能不满意，可以安装自己编译的triton=3.2.0，该版本，可能对性能有略微的提升。
请查看下面额外的triton相关的编译安装教程，使用下面的命令替换triton为自己编译的版本
cd triton-3.2.0-mi50/python/dist/
pip install triton-3.2.0-cp310-cp310-linux_x86_64.whl --force-reinstall
```

Updated:20250404----以下是vllm安装教程，mlc的不需要看，如果需要编译triton，可以只看triton部分
# Installation
请使用Ubuntu 22.04 多人测试24版本很多报错

建议使用conda作为虚拟环境
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
conda init
conda --version
conda config --set auto_activate_base ture
conda create --name vllmnew python=3.10
conda activate vllmnew
```

魔改最新版Triton 3.2.0安装指南 推荐使用python3.10这样的话跟我的测试环境完全一样

有问题可以在issue区留言或者进qq群:1028429001 ->本地部署deepseek-r1:671b纯cpu方案

本介绍基于https://github.com/Said-Akbar/triton-gcn5/blob/gcn5-edits/README.md (需要triton3.1.0可以去这里自取)

请注意,不要尝试rocm 6.2.2之外的版本, triton与vllm版本依赖严重, 6.3.3是测试版尝试过全是报错.

This is a fork of triton for AMD MI25/50/60.
I assume you already have rocm 6.2.2 installed with GPU drivers.
If not, use these commands (assuming you have Ubuntu 22.04):

Install rocm 6.2.2

AMD官方rocm安装说明(不建议使用官方的，因为他现在默认是6.3.3) https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html
```
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
wget https://repo.radeon.com/amdgpu-install/6.2.2/ubuntu/jammy/amdgpu-install_6.2.60202-1_all.deb
sudo apt install ./amdgpu-install_6.2.60202-1_all.deb
sudo apt update
sudo apt install amdgpu-dkms rocm
# linker
sudo tee --append /etc/ld.so.conf.d/rocm.conf <<EOF
/opt/rocm/lib
/opt/rocm/lib64
EOF
sudo ldconfig

# path
export PATH=$PATH:/opt/rocm-6.2.2/bin

# verify drivers
dkms status

# You need to close all windows and reboot to make changes effective.
reboot
```

Now you have rocm 6.2.2.

Install triton 3.2.0.

```
# create venv
#如果你不想使用python venv可以使用conda代替python3 -m venv vllmenv
#conda create --name vllmnew  python=3.10
#conda activate vllmnew

python3 -m venv vllmenv
source vllmenv/bin/activate

git clone https://github.com/gengchaogit/triton-3.2.0-mi50.git
cd triton-3.2.0-mi50
#这一步非常关键 本次改动只针对release最新版3.2.0
git checkout release/3.2.x
cd python
# install triton req’s
pip3 install ninja cmake wheel pybind11
# install torch first
pip3 install --no-cache-dir --pre torch>=2.6 torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2
# build triton
# 这里注意，必须要构建whl包，因为安装中的失败、重新安装依赖被覆盖等原因，我们可能需要最后安装完成的时候重新装一遍这个whl，一次构建多次安装
pip wheel . -w dist/
# 生成的文件将会出现在triton-3.2.0-mi50/python/dist/triton-3.2.0-cp310-cp310-linux_x86_64.whl (因为我是python3.10所以这里是310，根据python不同版本可能是311也可能是312)
cd dist
pip install triton-3.2.0-cp310-cp310-linux_x86_64.whl --force-reinstall
# 将来如果重新安装triton-3.2.0的话只需重新cd dist & triton-3.2.0-cp310-cp310-linux_x86_64.whl --force-reinstall
# 验证安装是否成功
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

预期输出
CUDA available: True

```

End of installation.

Triton3.2.0 编译安装教程结束，现在你可以编译最新版本的vllm了

下面是安装最新版vllm教程 或者你可以自己去看官方文档，传送门 https://github.com/vllm-project/vllm
```
# Clone git
git clone https://github.com/vllm-project/vllm.git
cd vllm
#如果你使用conda 使用下面命令代替source vllmenv/bin/activate
#conda activate vllmnew

source vllmenv/bin/activate
pip install --upgrade pip

# Build & install AMD SMI
pip install /opt/rocm/share/amd_smi

# Install dependencies
pip install --upgrade numba scipy huggingface-hub[cli,hf_transfer] setuptools_scm
pip install "numpy<2"
pip install -r requirements/rocm.txt

# Build vLLM for MI50.
export PYTORCH_ROCM_ARCH="gfx906"
python3 setup.py develop --verbose
#如果遇到cmake 3.5的问题，就去那个文件里开头第十几行把小于3.5的3.x版本改成3.5保存即可

#如果本次安装某些地方安装依赖覆盖掉了魔改版的triton需要你返回dist重新安装编译好的triton
#pip install triton-3.2.0-cp310-cp310-linux_x86_64.whl --force-reinstall

# vllm amd双卡启动命令-随便写的
# ps:可以接入dify/openwebui等支持openai接口的前端进行测试
VLLM_USE_TRITON_FLASH_ATTN=1 ROCM_PATH=/opt/rocm TORCH_BLAS_PREFER_HIPBLASLT=0 PYTORCH_ROCM_ARCH=gfx906 vllm serve /data1/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf --port 8099 --max-model-len 4096 --tensor-parallel-size 2 --served-model-name vllm

#若conda环境启动遇到动态链接库问题
conda install -c conda-forge libstdcxx-ng
#预期输出
(vllmnew) root@epyc:~# strings /root/miniconda3/envs/vllmnew/lib/libstdc++.so.6|grep GLIBCXX_3.4.30
GLIBCXX_3.4.30
GLIBCXX_3.4.30

```



下面的可以不用看，原始的官方readme

<div align="center">
  <img src="https://lh5.googleusercontent.com/wzQKEsTFkrgNQO9JjhGH5wFvslJr1saLtLaJ_a6Fp_gNENpvt3VG7BmztwngU9hFJaU4CPwGiw1opQtDvTkLrxWRbO_a12Q-pdESWHgtmheIHcPbOL5ZMC4TSiJVe5ty1w=w3517" alt="Triton logo">
</div>

| **`Documentation`** | **`Nightly Wheels`** |
|-------------------- | -------------------- |
| [![Documentation](https://github.com/triton-lang/triton/actions/workflows/documentation.yml/badge.svg)](https://triton-lang.org/) | [![Wheels](https://github.com/triton-lang/triton/actions/workflows/wheels.yml/badge.svg)](https://github.com/triton-lang/triton/actions/workflows/wheels.yml) |

# Triton

This is the development repository of Triton, a language and compiler for writing highly efficient custom Deep-Learning primitives. The aim of Triton is to provide an open-source environment to write fast code at higher productivity than CUDA, but also with higher flexibility than other existing DSLs.

The foundations of this project are described in the following MAPL2019 publication: [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf). Please consider citing this work if you use Triton!

The [official documentation](https://triton-lang.org) contains installation instructions and tutorials.  See also these third-party [Triton puzzles](https://github.com/srush/Triton-Puzzles), which can all be run using the Triton interpreter -- no GPU required.

# Quick Installation

You can install the latest stable release of Triton from pip:

```shell
pip install triton
```

Binary wheels are available for CPython 3.9-3.13.

# Enabling Blackwell Support

The main branch now features support for NVIDIA Blackwell GPUs using 5th
generation tensor cores. To enable this, you will need two additional steps:

1. Build a pre-release PyTorch from source with CUDA 12.8
2. Build triton from the latest source


First, to build pytorch you need to have CUDA 12.8 installed locally. If not,
follow the [instructions for your platform](https://developer.nvidia.com/cuda-downloads)
```bash
# Clone and checkout pytorch 2.6 release candidate
git clone https://github.com/pytorch/pytorch
cd pytorch
git checkout v2.6.0-rc9
git submodule sync
git submodule update --init --recursive -j 8

# Install build dependencies (assumes you already have a system compiler)
pip install -r requirements.txt
pip install mkl-static mkl-include wheel

# Build PyTorch (will take a long time)
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=$CUDA_HOME
export TORCH_CUDA_ARCH_LIST=Blackwell
python setup.py develop

# Optional, package build into a wheel to install on other machines.
python setup.py bdist_wheel
ls dist  # Wheel should be output in this directory
```

Note that if you use the domain libraries (`torchvision`, `torchtext`,
`torchaudio`, etc.) these will need to be built from source as well, otherwise
their custom PyTorch extensions will not work.

Finally, follow the instructions below to install triton from source.

# Install from source

```shell
git clone https://github.com/triton-lang/triton.git
cd triton

pip install -r python/requirements.txt # build-time dependencies
pip install -e python
```

Or with a virtualenv:

```shell
git clone https://github.com/triton-lang/triton.git
cd triton

python -m venv .venv --prompt triton
source .venv/bin/activate

pip install -r python/requirements.txt # build-time dependencies
pip install -e python
```

# Building with a custom LLVM

Triton uses LLVM to generate code for GPUs and CPUs.  Normally, the Triton build
downloads a prebuilt LLVM, but you can also build LLVM from source and use that.

LLVM does not have a stable API, so the Triton build will not work at an
arbitrary LLVM version.

1. Find the version of LLVM that Triton builds against.  Check
`cmake/llvm-hash.txt` to see the current version. For example, if it says:
       49af6502c6dcb4a7f7520178bd14df396f78240c

   This means that the version of Triton you have builds against
   [LLVM](https://github.com/llvm/llvm-project) 49af6502.

2. `git checkout` LLVM at this revision.  Optionally, make additional
   modifications to LLVM.

3. [Build LLVM](https://llvm.org/docs/CMake.html).  For example, you might run

       $ cd $HOME/llvm-project  # your clone of LLVM.
       $ mkdir build
       $ cd build
       $ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"
       $ ninja

4. Grab a snack, this will take a while.

5. Build Triton as above, but set the following environment variables.

       # Modify as appropriate to point to your LLVM build.
       $ export LLVM_BUILD_DIR=$HOME/llvm-project/build

       $ cd <triton install>
       $ LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
         LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
         LLVM_SYSPATH=$LLVM_BUILD_DIR \
         pip install -e python

# Tips for building

- Set `TRITON_BUILD_WITH_CLANG_LLD=true` as an environment variable to use clang
  and lld.  lld in particular results in faster builds.

- Set `TRITON_BUILD_WITH_CCACHE=true` to build with ccache.

- Set `TRITON_HOME=/some/path` to change the location of the `.triton`
  directory where Triton's cache is located and downloads are stored
  during the build. By default, this is the user's home directory. It
  can be changed anytime.

- If you're running out of memory when building Triton, specify the `MAX_JOBS`
  environment variable (to the `pip install -e python` command) to limit the
  number of jobs.

- Pass `--no-build-isolation` to `pip install` to make nop builds faster.
  Without this, every invocation of `pip install` uses a different symlink to
  cmake, and this forces ninja to rebuild most of the `.a` files.

- vscode intellisense has some difficulty figuring out how to build Triton's C++
  (probably because, in our build, users don't invoke cmake directly, but
  instead use setup.py).  Teach vscode how to compile Triton as follows.

    - Do a local build. Run command `pip install -e python`
    - Get the full path to the `compile_commands.json` file produced by the build:
      `find python/build -name 'compile_commands.json' | xargs readlink -f`.
      You might get a full path similar to `/Users/{username}/triton/python/build/cmake.macosx-11.1-arm64-cpython-3.12/compile_commands.json`
    - In vscode, install the
      [C/C++
      extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools),
      then open the command palette (`Shift + Command + P` on Mac, or `Shift +
      Ctrl + P` on Windows/Linux) and open `C/C++: Edit Configurations (UI)`.
    - Open "Advanced Settings" and paste the full path to
      `compile_commands.json` into the "Compile Commands" textbox.

# Running tests

There currently isn't a turnkey way to run all the Triton tests, but you can
follow the following recipe.

```shell
# One-time setup.  Note this will reinstall local Triton because torch
# overwrites it with the public version.
$ make dev-install

# To run all tests (requires a GPU)
$ make test

# Or, to run tests without a gpu
$ make test-nogpu
```

# Tips for hacking

For detailed instructions on how to debug Triton's frontend, please refer to this [tutorial](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html). The following includes additional tips for hacking on Triton's backend.

**Helpful environment variables**

- `MLIR_ENABLE_DUMP=1` dumps the IR before every MLIR pass Triton runs, for all
   kernels. Use `MLIR_ENABLE_DUMP=kernelName` to dump for a specific kernel only.
  - Triton cache can interfere with the dump. In cases where `MLIR_ENABLE_DUMP=1` does not work, try cleaning your triton cache: `rm -r ~/.triton/cache/*`
- `MLIR_DUMP_PATH` specifies where `MLIR_ENABLE_DUMP` will dump to. If unset will dump to stderr.
- `LLVM_IR_ENABLE_DUMP=1` dumps the IR before every pass run over the LLVM IR.
- `TRITON_REPRODUCER_PATH=<reproducer_path>` will generate an MLIR reproducer file
  at `<reproducer_path>` before each MLIR compiler stage. If any of the stages fail,
  `<reproducer_path>` will be a local MLIR reproducer captured right before the failing pass.
- `TRITON_INTERPRET=1` uses the Triton interpreter instead of running on the
  GPU.  You can insert Python breakpoints in your kernel code!
- `TRITON_ENABLE_LLVM_DEBUG=1` passes `-debug` to LLVM, printing a lot of
  debugging information to stdout.  If this is too noisy, run with just
  `TRITON_LLVM_DEBUG_ONLY` instead to limit the output.

  An alternative way to reduce output noisiness is running with
  `LLVM_IR_ENABLE_DUMP=1`, extract the IR before the LLVM pass of interest, and
  then run LLVM's `opt` standalone, perhaps passing `-debug-only=foo` on the
  command line.
- `TRITON_LLVM_DEBUG_ONLY=<comma-separated>` is the equivalent of LLVM's
  `-debug-only` command-line option. This limits the LLVM debug output to
  specific pass or component names (which are specified using `#define
  DEBUG_TYPE` throughout LLVM and Triton) in order to allow the debug output to
  be less noisy. `TRITON_LLVM_DEBUG_ONLY` allows for one or more comma
  separated values to be specified (eg
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions"` or
  `TRITON_LLVM_DEBUG_ONLY="tritongpu-remove-layout-conversions,regalloc"`).
- `TRITON_ENABLE_ASAN=1` invokes the LLVM address sanitizer for
  memory leak and out of bounds access detection. Currently only supported on the AMD
  backend. This must be run using the ASAN libraries documented [here](https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/using-gpu-sanitizer.html).

  When enabling the address sanitizer it is recommended to disable various memory caching strategies
  both within the ROCm stack and PyTorch. This will give the address sanitizer the best chance at finding the
  memory fault where it originates. See this [test](https://github.com/triton-lang/triton/blob/main/third_party/amd/python/test/test_address_sanitizer.py) for more details.

- `USE_IR_LOC={ttir,ttgir}` reparses the IR such that the location information
  will be the line number of the IR file with that particular extension,
  instead of line number of the python file. This can provide a direct mapping
  from the IR to llir/ptx. When used with performance tools, it can provide a
  breakdown on IR instructions.
- `TRITON_PRINT_AUTOTUNING=1` prints out the best autotuning config and total time
  spent for each kernel after autotuning is complete.
- `DISABLE_LLVM_OPT` will disable llvm optimizations for make_llir and make_ptx
  if its value is true when parsing as Bool. Otherwise, it will be parsed as a list
  of flags to disable llvm optimizations. One usage case is
  `DISABLE_LLVM_OPT="disable-lsr"`
  Loop strength reduction is known to cause up to 10% performance changes for
  certain kernels with register pressure.
- `TRITON_ALWAYS_COMPILE=1` forces to compile kernels regardless of cache hit.
- `MLIR_ENABLE_TIMING` dumps the timing information for each MLIR pass.
- `LLVM_ENABLE_TIMING` dumps the timing information for each LLVM pass.
- `TRITON_DEFAULT_FP_FUSION` overrides the default behavior of allowing fp fusion (mul+add->fma).
- `MLIR_ENABLE_DIAGNOSTICS=<comma-separated>` controls diagnostic emission in MLIR.
  Options are: `warnings`, `remarks`, `stacktraces`, `operations`.
  Use comma-separated values to customize output. For example,
  `MLIR_ENABLE_DIAGNOSTICS=remarks,operations` enables remarks and IR operations,
  while `MLIR_ENABLE_DIAGNOSTICS=warnings,stacktraces` enables warnings with
  stacktraces. By default, only errors are shown. Setting `warnings` includes
  errors and warnings; `remarks` includes errors, warnings, and remarks.
- `MLIR_ENABLE_REMARK` is deprecated. Please use `MLIR_ENABLE_DIAGNOSTICS=remarks`.
- `TRITON_KERNEL_DUMP` enables the dumping of the IR from each compilation stage and the final ptx/amdgcn.
- `TRITON_DUMP_DIR` specifies the directory to save the dumped IR and ptx/amdgcn when `TRITON_KERNEL_DUMP` is set to 1.
- `TRITON_KERNEL_OVERRIDE` enables the override of the compiled kernel with a user-specified IR/ptx/amdgcn at the beginning of each compilation stage.
- `TRITON_OVERRIDE_DIR` specifies the directory from which to load the IR/ptx/amdgcn files when `TRITON_KERNEL_OVERRIDE` is set to 1.
- `TRITON_F32_DEFAULT` sets the default input precision of `tl.dot` when using 32-bit floats, which can be either `ieee`, `tf32`, or `tf32x3`.
- `TRITON_FRONT_END_DEBUGGING=1` disables exception wrapping when an error occurs in the compiler frontend, allowing the full stack trace to be seen.

**Kernel Override Steps**

```bash
export TRITON_ALWAYS_COMPILE=1
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=<dump_dir>
export TRITON_KERNEL_OVERRIDE=1
export TRITON_OVERRIDE_DIR=<override_dir>
# Step 1: Run the kernel once to dump kernel's IRs and ptx/amdgcn in $TRITON_DUMP_DIR
# Step 2: Copy $TRITON_DUMP_DIR/<kernel_hash> to $TRITON_OVERRIDE_DIR
# Step 3: Delete the stages that you do not want to override and modify the stage you do want to override
# Step 4: Run the kernel again to see the overridden result
```


# Changelog

Version 2.0 is out! New features include:

- Many, many bug fixes
- Performance improvements
- Backend rewritten to use MLIR
- Support for kernels that contain back-to-back matmuls (e.g., flash attention)

# Contributing

Community contributions are more than welcome, whether it be to fix bugs or to add new features at [github](https://github.com/triton-lang/triton/). For more detailed instructions, please visit our [contributor's guide](CONTRIBUTING.md).

# Compatibility

Supported Platforms:

- Linux

Supported Hardware:

- NVIDIA GPUs (Compute Capability 8.0+)
- AMD GPUs (ROCm 6.2+)
- Under development: CPUs

# Development Container (Dev Container)

**Dev Containers** for the Triton project are available from
the [triton-dev-containers repository](https://github.com/redhat-et/triton-dev-containers)

### Key Benefits:
- **Consistency**: All developers can work with the same development
  environment, ensuring uniform behavior across different systems.
- **Isolation**: The container prevents potential conflicts with software
  installed on your local machine.
- **Portability**: Easily share the development environment with team members,
  minimizing onboarding time and setup issues.

### How to Use the Dev Container:

For detailed instructions on how to use the dev containers please see
the [dev container user guide](https://github.com/redhat-et/triton-dev-containers/blob/main/.devcontainer/devcontainer.md)
