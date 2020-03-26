---
layout: post
title:  "fastai2 in Singularity Container"
description: How I built a Singularity container with editable fastai2 installation for multi-user HPC environment.
date:   2020-03-26 18:00:00 +0
categories: fastai DL singularity containerisation
---
The awesome people at [fast.ai][fastai] started the 2020 iteration (aka '[part1-v4][v4]') of their wildly popular [Deep Learning Part I][usf] course earlier this month, running it entirely online because of [covid-19][fa-covid] social (distancing) responsibility.

The course is using the brand new fastai v2 library ([fastai2][fastai2], currently in pre-release) along with [PyTorch][pytorch], and makes a start in covering the content of their upcoming [book][oreilly].

Installation of the fastai v2 library can be pretty straightforward using `conda` and `pip`. It is also well-supported on various cloud GPU platforms such as [Paperspace][paperspace] and [Colab][colab]. However, as with many other cutting-edge deep learning software stacks that typically involve quite frequent updates and changes (for bugfixes, performance enhancements, etc.), it can be a challenge to have everything setup in a multi-user [HPC][hpc] environment, without the risk of affecting other users' software packages needed for production work.

[Containerisation][wiki] technology presents a possible solution to these challenges, by enabling self-contained (hah!) containers that can be built and deployed with all the internally consistent dependencies, without affecting other parts of the host system or other containers. [Docker][docker] is arguably the most well-known container system right now, but it might not necessarily be the best for a multi-user HPC environment used for projects and production —instead of experimentation— work, as it can be difficult to setup and ensure the correct user/group permissions in the host system are replicated and honoured in Docker containers. There also seems to be potential risk of undesired [privilege escalation][root] to `root` access due to the way that the Docker daemon works, which is again a problem for multi-user production HPC.

My quick search showed that a different container system, [Singularity][singularity], might be better-suited for my use case above. The article [here][pythonspeed] helpfully describes some of the problems in Docker defaults that can be solved by Singularity. Even though I do not have `sudo` permission on the multi-user HPC, I am able to build Singularity containers with [fastai2][fastai2] on a different machine (where I have `sudo`), e.g. a cheap and cheerful small cloud instance. And when I (and/or others) run the container on the HPC, it will [natively support][gpu] NVIDIA's CUDA GPU compute for deep learning, honour user/group permissions and filesystem access on the HPC, and will not break or interfere with other software stacks (e.g. [finite element analysis][dyna] with [MPI][mpi], and GPU-enabled [CFD][cfd] with a different [CUDA][cuda] version) on the HPC used by other users. This gives me the flexibility of being able to experiment and tinker with the latest development version of [fastai2][fastai2] (or other deep learning packages) without having `sudo` on the HPC, prepare and share Singularity containers that have functioning [fastai2][fastai2] installations, while retaining the rigidity and stability needed for existing software with potentially conflicting dependencies and project-based user security permissions on the HPC.

I have not been experimenting with and using Singularity containers for very long yet, but I will try to describe the steps I took to build the Singularity container with an editable install (i.e. linked to an update-able Git repository) of [fastai2][fastai2].

### Installing Singularity

Firstly, Singularity will need to be installed by the sysadmin on the HPC by just following the [installation guide][install]. If a separate machine/instance is used to build the Singularity containers (like in my case), then Singularity needs to be installed there too, and `root` permission is needed for the container-build.

### Creating Singularity _def_ file

Next, a Singularity [definition file][def] (similar to Docker's _Dockerfile_) is created, to have all the steps needed to build the container with the software ([fastai2][fastai2] in this example) and its dependencies (e.g. [fastai v1 library][fastaiv1], [fastcore][fastcore], etc.), plus any ancillaries (e.g. [Jupyter Notebook][jupyter]).

Singularity containers can be bootstrapped from Docker images (which are more popular and widely available), and so in the _def_ file I [start][header] with NVIDIA's own [Docker image containing CUDA][nvidia-cuda]:
```
BootStrap: docker
From: nvidia/cuda
```

Then, define the [environment][env] variables that will be set at runtime (i.e. when the container is used):
```sh
%environment
    export LANG=C.UTF-8
    export PATH=$PATH:/opt/conda/bin
    export PYTHON_VERSION=3.7
    export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
```

The next bit contains the steps that will be used to install [fastai2][fastai2] and its dependencies, within the [`%post`][post] section of the _def_ file. Again, start by defining the same environment variables, which are used also at build-time (as opposed to _runtime_, mentioned above):
```sh
%post
    export LANG=C.UTF-8
    export PATH=$PATH:/opt/conda/bin
    export PYTHON_VERSION=3.7
    export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
```

Then, install the software and tools needed to setup [fastai2][fastai2] later on. The default OS in NVIDIA's CUDA Docker image is Ubuntu, and so `apt-get` is used for this step. I also update `pip`, and install [`miniconda`][miniconda], as `conda` will be used in the next step.
```sh
    apt-get -y update
    apt-get -y install --no-install-recommends build-essential ca-certificates \ 
            git vim zip unzip curl python3-pip python3-setuptools graphviz
    apt-get clean
    rm -rf /var/lib/apt/lists/*

    pip3 install --upgrade pip

    curl -o ~/miniconda.sh -O \
      https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      && chmod +x ~/miniconda.sh \
      && ~/miniconda.sh -b -p /opt/conda \
      && rm ~/miniconda.sh \
      && conda install conda-build
```

Next, go ahead and use `conda` to install [fastai v1 library][fastaiv1], and while we are at it, also install [Jupyter Notebook][jupyter] and its extensions:
```sh
    conda update conda && conda install -c pytorch -c fastai fastai \
      && conda install jupyter notebook \
      && conda install -c conda-forge jupyter_contrib_nbextensions
```

As I am going to do an editable `pip` install of both [fastai2][fastai2] and the [fastcore][fastcore] dependency, I `git clone` the two repositories. Note that they are cloned into a shared filepath that exists on the HPC host system, so that I can choose to `git pull` update the repositories **on the HPC host** in future, and all the user(s) running the [fastai2][fastai2] Singularity container will automatically pick up the latest updates on the editable install:
```sh
    mkdir -p /data/shared
    cd /data/shared && git clone https://github.com/fastai/[fastai2][fastai2] \
    && git clone https://github.com/fastai/fastcore
```

Then, run the editable `pip` installs, as recommended currently by fastai as "_probably the best approach at the moment, since fastai v2 is under heavy development_" still:
```sh
    cd /data/shared/fastcore && python3.7 -m pip install -e ".[dev]"
    cd /data/shared/[fastai2][fastai2]  && python3.7 -m pip install -e ".[dev]"
```

As a final setup step, install some other libraries and packages used in the [part1-v4][v4] fastai course:
```sh
    conda install pyarrow
    python3.7 -m pip install graphviz ipywidgets matplotlib nbdev>=0.2.12 \
        pandas scikit_learn azure-cognitiveservices-search-imagesearch sentencepiece
```

With that, all the necessary installs and setup should be there. I then add the '[start script][start]' that will be executed when the Singularity container is started. In this case:
* Start the Jupyter Notebook server
* Make it accessible to other computers/IP (firewalled to internal network only, in our case)
* Have the server listen to a non-default port of 9999 (Jupyter default is 8888)
* Give it a password hash for access (in this case, the hash corresponds to password _fastai_)
* Make it start in the shared filepath on the HPC host system where I cloned the [fastai2][fastai2] and [fastcore][fastcore]  repositories. This is also where I have other shared files needed (e.g. the [part1-v4][v4] course material)

```
%startscript
    jupyter notebook --ip=0.0.0.0 --port=9999 --no-browser \
        --NotebookApp.password='sha1:a60ff295d0b9:506732d050d4f50bfac9b6d6f37ea6b86348f4ed' \
        --log-level=WARN --notebook-dir=/data/shared/ &
```

Finish the _def_ file by adding some basic label and descriptions:
```
%labels
    ABOUT container for fastai2 (dev editable install) with jupyter notebook on startup (port 9999), for March 2020 fastai course
    AUTHOR Yijin Lee
```

The complete example _def_ file explained above can be found [here][repo].

### Building the Singularity container

With the _def_ file, I can now build the Singularity container to get the resulting container _sif_ file. I needed `sudo` or `root` permission for this, and so I used a cheap AWS instance (t2.small), instead of the HPC environment (where I only have basic user permissions). My AWS instance only has limited `/` root device file space, and so I set an environment variable for Singularity to use a different AWS block device storage as the temp directory (or else the build will fail):
```bash
root@aws-t2:~# export TMPDIR=/blockdevice/tmp
root@aws-t2:~# ls
fastai2.def
root@aws-t2:~# singularity build fastai2.sif fastai2.def
```

With the Singularity build, the requested _sif_ file will be created. It is quite a big file, at around 5.0GB, but I only really needed to build and transfer it once, since it will contain an editable (and thus update-able) install of [fastai2][fastai2]:
```bash
root@aws-t2:~# ls -lh
-rw-r--r-- 1 root   root   1.9K Mar 25 12:00 fastai2.def
-rwxr-xr-x 1 root   root   5.0G Mar 25 12:30 fastai2.sif
```

The _sif_ file can then be copied/transferred to the HPC environment for actual use.

### Running the Singularity container

As I want to use NVIDIA GPU for deep learning compute, the HPC where I run the [fastai2][fastai2] Singularity container needs to have the correct NVIDIA GPU [drivers][driver] installed (by the sysadmin). Note that the only hard requirement is the drivers — CUDA and other dependencies are self-contained in our Singularity _sif_ file already, all with the correct versions. I can check the NVIDIA GPU status by running `nvidia-smi`:
```sh
[ylee@hpc01 shared]$ nvidia-smi
Thu Mar 25 13:00:00 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.33.01    Driver Version: 440.33.01    CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-PCIE...  Off  | 00000000:37:00.0 Off |                    0 |
| N/A   53C    P0    30W / 250W |     14MiB / 16160MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      2468      G   Xorg                                          14MiB |
+-----------------------------------------------------------------------------+
```

I will start the Singularity container from the shared filepath where the necessary files (e.g. [fastai2][fastai2] and [fastcore][fastcore] repositories, [part1-v4][v4] course material, etc.) reside — this was mentioned above. In my case, this is in `/data/shared`, and my _sif_ file is in `/data/shared/singularity`:
```sh
[ylee@hpc01 shared]$ pwd
/data/shared
[ylee@hpc01 shared]$ singularity instance start --nv ./singularity/fastai2.sif fastai2
INFO:    instance started successfully
[ylee@hpc01 shared]$ singularity instance list
INSTANCE NAME    PID      IMAGE
fastai2          13579    /data/shared/singularity/fastai2.sif
```

The `--nv` flag above is for [Singularity][sifgpu] to be able to [leverage NVIDIA GPU][sifgpu2].

Because of the 'startscript' defined in the _def_ file, there should now be a Jupyter Notebook server running and listening on port 9999:
```sh
[ylee@hpc01 shared]$ netstat -plunt
(Not all processes could be identified, non-owned process info
 will not be shown, you would have to be root to see it all.)
Active Internet connections (only servers)
Proto Recv-Q Send-Q Local Address           Foreign Address         State       PID/Program name
tcp        0      0 0.0.0.0:9999            0.0.0.0:*               LISTEN      13579/python
```

I can thus point a web browser to the IP at port 9999, and enter the password (defined as _fastai_ in the hash within our _def_ file) to access Jupyter Notebook.

I can also run a [shell][sifshell] within the Singularity container instance, to start interactive Python directly, without going via Jupyter Notebook:
```sh
[ylee@hpc01 shared]$ singularity shell instance://fastai2
Singularity fastai2.sif:/data/shared> python3.7
Python 3.7.6 (default, Jan  8 2020, 19:59:22)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
```

From within Python, I can also quickly confirm that [fastai2][fastai2] is indeed installed, and CUDA compute is available for [PyTorch][pytorch]:
```python
>>> from fastai2.vision.all import *
>>> torch.cuda.is_available()
True
```

And, because the container has an editable `pip` install of [fastai2][fastai2] residing on the HPC host system, I can `git pull` or `git checkout` to a specific [fastai2][fastai2] commit from the HPC, and all users of the Singularity container will then 'get' the corresponding [fastai2][fastai2] version. For example, starting with a slightly older version (0.0.14):
```python
>>> import fastai2
>>> fastai2.__version__
...
'0.0.14'
>>> exit()
```

I can exit from the Singularity instance shell to get back to the HPC host system, while leaving the container still running. I then change the [fastai2][fastai2] version (e.g. update to the latest via `git pull`), and the change will be 'live' back in the Singularity instance shell.
```sh
Singularity fastai2.sif:/data/shared> exit
exit
[ylee@hpc01 shared]$ cd fastai2
[ylee@hpc01 fastai2]$ git pull
.
.
.
[ylee@hpc01 fastai2]$ cd ..
[ylee@hpc01 shared]$ singularity shell instance://fastai2
Singularity fastai2.sif:/data/shared> python3.7
Python 3.7.6 (default, Jan  8 2020, 19:59:22)
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import fastai2
>>> fastai2.__version__
'0.0.16'
```

All the shared filesystem files (e.g. _ipynb_ notebooks) can be accessed from within the container, retaining the original user/group permissions, without having to do/set anything for Singularity. When done, just stop the running container:
```sh
[ylee@hpc01 shared]$ singularity instance stop fastai2
Killing fastai2 instance of /data/shared/singularity/fastai2.sif (PID=13579) (Timeout)
```

### Summary

Without getting `sudo` or `root` permission on a production HPC cluster, I can define and build a Singularity container on a separate cheap cloud instance (where `root` is available), which can have a `pip` editable install of [fastai2][fastai2].

The resulting container _sif_ file can be used on the HPC cluster, have native access to GPU CUDA compute, easily retain user/group permissions in the multi-user HPC environment, and have all the necessary software stack dependencies (except NVIDIA GPU driver, which must be present on the HPC host system) without messing up or interfering with other software stacks or environments on the HPC host system.

The editable install residing on the HPC host filesystem means that I can easily upgrade/change the version of [fastai2][fastai2] via `git`, and users of the Singularity container can get the corresponding version changes 'live'. This allows a 'balance' of having flexibility to experiment with software stacks in a multi-user production HPC environment with native user/group permissions, while reducing the risks of messing things up for everyone (e.g. via undesired `root` privilege escalation that can happen in Docker). It also means that other users can all re(use) the same container with the same versions of software stack, e.g. for a fastai study group.

My Singularity example _def_ file explained above can be found [here][repo]. And please do join us for lively discussions on the [fastai forums][forum].

[fastai]: https://www.fast.ai
[v4]: https://github.com/fastai/course-v4
[usf]: https://www.usfca.edu/data-institute/certificates/deep-learning-part-one
[fa-covid]: https://www.fast.ai/2020/03/09/coronavirus/
[fastai2]: https://github.com/fastai/fastai2
[pytorch]: https://pytorch.org/
[oreilly]: https://www.oreilly.com/library/view/deep-learning-for/9781492045519/
[paperspace]: https://www.paperspace.com/
[colab]: https://colab.research.google.com/
[hpc]: https://en.wikipedia.org/wiki/High-performance_computing
[wiki]: https://en.wikipedia.org/wiki/OS-level_virtualization
[docker]: https://www.docker.com
[root]: https://www.hackingarticles.in/docker-privilege-escalation/
[singularity]: https://sylabs.io/singularity/
[pythonspeed]: https://pythonspeed.com/articles/containers-filesystem-data-processing/
[gpu]: https://sylabs.io/guides/3.5/user-guide/gpu.html
[dyna]: https://www.arup.com/dyna
[mpi]: https://en.wikipedia.org/wiki/Message_Passing_Interface
[cfd]: https://en.wikipedia.org/wiki/Computational_fluid_dynamics
[cuda]: https://developer.nvidia.com/about-cuda
[install]: https://sylabs.io/guides/3.5/user-guide/quick_start.html#quick-installation-steps
[def]: https://sylabs.io/guides/3.5/user-guide/definition_files.html
[fastaiv1]: https://github.com/fastai/fastai
[fastcore]: https://github.com/fastai/fastcore
[jupyter]: https://jupyter.org/
[nvidia-cuda]: https://hub.docker.com/r/nvidia/cuda
[header]: https://sylabs.io/guides/3.5/user-guide/definition_files.html#header
[env]: https://sylabs.io/guides/3.5/user-guide/definition_files.html#environment
[post]: https://sylabs.io/guides/3.5/user-guide/definition_files.html#post
[miniconda]: https://docs.conda.io/en/latest/miniconda.html
[start]: https://sylabs.io/guides/3.5/user-guide/definition_files.html#startscript
[repo]: https://github.com/yijinlee/fastai2-def
[driver]: https://www.nvidia.com/Download/index.aspx
[sifgpu]: https://sylabs.io/guides/3.5/user-guide/gpu.html#nvidia-gpus-cuda
[sifgpu2]: https://docs.nvidia.com/ngc/ngc-user-guide/singularity.html#running-the-singularity-container
[sifshell]: https://sylabs.io/guides/3.5/user-guide/cli/singularity_shell.html
[forum]: https://forums.fast.ai/t/fastai2-in-singularity-container/66727
