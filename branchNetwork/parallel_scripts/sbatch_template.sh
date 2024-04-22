Last login: Fri Apr 12 14:23:54 on ttys000
(base) matthew@d141-109 ~ % ssh mtrappet@talapas.login.uoregon.edu
ssh: Could not resolve hostname talapas.login.uoregon.edu: nodename nor servname provided, or not known
(base) matthew@d141-109 ~ % ssh mtrappet@talapas.uoregon.edu 
ssh: Could not resolve hostname talapas.uoregon.edu: nodename nor servname provided, or not known
(base) matthew@d141-109 ~ % ssh mtrappet@login.talapas.uoregon.edu
(mtrappet@login.talapas.uoregon.edu) Password: 
(mtrappet@login.talapas.uoregon.edu) Duo two-factor login for mtrappet

Enter a passcode or select one of the following options:

 1. Duo Push to XXX-XXX-3550
 2. Duo Push to iPad Pro - Matt (iOS)
 3. Phone call to XXX-XXX-3550
 4. SMS passcodes to XXX-XXX-3550

Passcode or option (1-4): 1
Success. Logging you in...
# Welcome to Talapas!
# Data on Talapas is NOT backed up. Data management is each users responsibility.
#
# Need support? Please visit the Talapas Knowledge Base:
# https://hpcrcf.atlassian.net/wiki/spaces/TW/overview
#
# You've connected to our *new* Talapas cluster.
# Please see the release notes for information on what's changed.
# https://hpcrcf.atlassian.net/l/cp/1AizPjHm
#
Last login: Mon Feb 26 12:07:02 2024 from 73.67.155.51
# Storage usage in GB as of Fri Apr 12 14:00:12 2024
Fileset          User             UsedByUser  UsedByAll      Quota  Use%
home             mtrappet                 22          -        250     9
tau              mtrappet                 43        568       2048    28
[mtrappet@login1 ~]$ sinfo
PARTITION      AVAIL  TIMELIMIT  NODES  STATE NODELIST
compute           up 1-00:00:00     38  drain n[0049-0085,0189]
compute           up 1-00:00:00     25    mix n[0013-0021,0023-0024,0028,0099-0102,0104-0106,0181-0182,0185-0188]
compute           up 1-00:00:00     18  alloc n[0022,0025-0027,0029-0036,0097-0098,0103,0107-0108,0193]
compute           up 1-00:00:00      9   idle n[0180,0183-0184,0190-0192,0194-0196]
computelong       up 14-00:00:0     26  drain n[0049-0072,0085,0189]
computelong       up 14-00:00:0     24    mix n[0013-0021,0023-0024,0099-0102,0104-0106,0181-0182,0185-0188]
computelong       up 14-00:00:0      6  alloc n[0022,0097-0098,0103,0107-0108]
computelong       up 14-00:00:0      6   idle n[0180,0183-0184,0190-0192]
gpu               up 1-00:00:00      1   plnd n0166
gpu               up 1-00:00:00     10    mix n[0153-0157,0164-0165,0167-0169]
gpu               up 1-00:00:00     11   idle n[0149-0151,0158-0160,0162-0163,0170-0172]
gpulong           up 14-00:00:0      1   plnd n0166
gpulong           up 14-00:00:0     10    mix n[0153-0157,0164-0165,0167-0169]
gpulong           up 14-00:00:0      4   idle n[0162-0163,0171-0172]
interactive       up   12:00:00     18  alloc n[0209-0212,0302-0313,0398-0399]
interactive       up   12:00:00      1   idle n0199
interactivegpu    up    8:00:00      1    mix n0301
interactivegpu    up    8:00:00      1   idle n0161
memory            up 1-00:00:00      6    mix n[0141,0144,0146,0148,0372,0378]
memory            up 1-00:00:00      3  alloc n[0142,0374,0376]
memory            up 1-00:00:00      7   idle n[0143,0145,0147,0373,0375,0377,0379]
memorylong        up 14-00:00:0      5    mix n[0144,0146,0148,0372,0378]
memorylong        up 14-00:00:0      3  alloc n[0142,0374,0376]
preempt           up 7-00:00:00      1   plnd n0166
preempt           up 7-00:00:00     38  drain n[0049-0085,0189]
preempt           up 7-00:00:00     77    mix n[0013-0021,0023-0024,0028,0037-0044,0099-0102,0104-0106,0141,0144,0146,0148,0152-0157,0164-0165,0167-0169,0181-0182,0185-0188,0213,0221,0230-0231,0246-0247,0301,0314-0316,0333-0336,0363,0365,0372,0378,0383,0389-0397,1000]
preempt           up 7-00:00:00    145  alloc n[0022,0025-0027,0029-0036,0097-0098,0103,0107-0108,0142,0173-0179,0193,0201-0212,0214-0220,0222-0229,0232-0245,0254-0265,0302-0313,0317-0332,0337-0358,0366-0371,0374,0376,0380-0382,0384,0386,0388,0398-0399]
preempt           up 7-00:00:00     41   idle n[0143,0145,0147,0149-0151,0158-0163,0170-0172,0180,0183-0184,0191-0192,0194-0196,0199,0248-0253,0359-0362,0364,0373,0375,0377,0379,0385,0387]
[mtrappet@login1 ~]$ srun -I -n 1 -N 1 -t 60 -p gpu --pty /bin/bash
srun: error: AssocGrpSubmitJobsLimit
srun: error: Unable to allocate resources: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
[mtrappet@login1 ~]$ srun -I -n 1 -N 1 -t 60 -p interactivegpu --pty /bin/bash
srun: error: AssocGrpSubmitJobsLimit
srun: error: Unable to allocate resources: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
[mtrappet@login1 ~]$ srun -I -n 1 -N 1 -t 30 -p interactivegpu --pty /bin/bash
srun: error: AssocGrpSubmitJobsLimit
srun: error: Unable to allocate resources: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)
[mtrappet@login1 ~]$ srun -I -n 1 -N 1 -t 30 -p interactivegpu --account=tau --pty /bin/bash
srun: error: Unable to allocate resources: Immediate execution impossible. Individual job submission scheduling attempts deferred
[mtrappet@login1 ~]$ srun -I -n 1 -N 1 -t 30 -p gpu --account=tau --pty /bin/bash
srun: error: Unable to allocate resources: Immediate execution impossible. Individual job submission scheduling attempts deferred
[mtrappet@login1 ~]$ salloc -I -n 1 -N 1 -t 30 -p gpu --account=tau --pty /bin/bash
salloc: unrecognized option '--pty'
Try "salloc --help" for more information
[mtrappet@login1 ~]$ salloc -I -n 1 -N 1 -t 30 -p gpu --account=tau
salloc: error: Job submit/allocate failed: Immediate execution impossible. Individual job submission scheduling attempts deferred
[mtrappet@login1 ~]$ ls
Desktop    Downloads  main.cc  ray_results  survival-grid-world  Templates
Documents  foo.ppk    Public   RaySlurm     tau                  testing
[mtrappet@login1 ~]$ cd RaySlurm/
[mtrappet@login1 RaySlurm]$ ls
examples           start-worker.sh            test.out
groupmeeting12_14  submit-ray-cluster.sbatch  use-ray-with-slurm
start-head.sh      test.err
[mtrappet@login1 RaySlurm]$ cd use-ray-with-slurm/
[mtrappet@login1 use-ray-with-slurm]$ ls
launch.py  sbatch_template.sh  test_1129-1508.log  test.py
python.sh  test_1129-1507.sh   test_1129-1508.sh
[mtrappet@login1 use-ray-with-slurm]$ nano test.py
[mtrappet@login1 use-ray-with-slurm]$ nano launch.py
[mtrappet@login1 use-ray-with-slurm]$ nano sbatch_template.sh 
[mtrappet@login1 use-ray-with-slurm]$ python launch.py -partition gpu
-bash: python: command not found
[mtrappet@login1 use-ray-with-slurm]$ ls
launch.py  sbatch_template.sh  test_1129-1508.log  test.py
python.sh  test_1129-1507.sh   test_1129-1508.sh
[mtrappet@login1 use-ray-with-slurm]$ source ~/
.bash_history        .gvfs/               .pdbhistory
.bash_logout         .ICEauthority        .pki/
.bash_profile        .imageio/            Public/
.bashrc              .ipython/            .python_history
.cache/              .java/               ray_results/
.conda/              .jupyter/            RaySlurm/
.condarc             .keras/              .ssh/
.config/             .kshrc               survival-grid-world/
.dbus/               .lmod.d/             tau/
Desktop/             .local/              Templates/
Documents/           main.cc              testing/
Downloads            .Mathematica/        .viminfo
.emacs               .mozilla/            .vnc/
.esd_auth            .mujoco/             .Xauthority
foo.ppk              .nv/                 
.gitconfig           .ParaProf/           
[mtrappet@login1 use-ray-with-slurm]$ source ~/survival-grid-world/
data-science/              optimize_rl_0131-1545.sh
data-science.yml           optimize_rl_0131-1708.log
.git/                      optimize_rl_0131-1708.sh
.gitignore                 README.md
hrl/                       requirements.txt
NoiseExperiments.ipynb     results/
optimize_rl_0131-1545.log  timing_29780948.out
[mtrappet@login1 use-ray-with-slurm]$ source ~/survival-grid-world/data-science/bin/activate
(data-science) [mtrappet@login1 use-ray-with-slurm]$ nano launch
(data-science) [mtrappet@login1 use-ray-with-slurm]$ nano launch.py
(data-science) [mtrappet@login1 use-ray-with-slurm]$ sbatch launch.py -p gpu
sbatch: error: This does not look like a batch script.  The first
sbatch: error: line must start with #! followed by the path to an interpreter.
sbatch: error: For instance: #!/bin/sh
(data-science) [mtrappet@login1 use-ray-with-slurm]$ python launch.py -p gpu
/home/mtrappet/RaySlurm/use-ray-with-slurm/sbatch_template.sh
Path(__file__): /gpfs/home/mtrappet/RaySlurm/use-ray-with-slurm/launch.py
usage: launch.py [-h] --exp-name EXP_NAME [--num-nodes NUM_NODES]
                 [--node NODE] [--num-gpus NUM_GPUS] [--partition PARTITION]
                 [--load-env LOAD_ENV] --command COMMAND
                 [--conda_env CONDA_ENV]
launch.py: error: the following arguments are required: --exp-name, --command
(data-science) [mtrappet@login1 use-ray-with-slurm]$ ls
launch.py  sbatch_template.sh  test_1129-1508.log  test.py
python.sh  test_1129-1507.sh   test_1129-1508.sh
(data-science) [mtrappet@login1 use-ray-with-slurm]$ nano python.sh
(data-science) [mtrappet@login1 use-ray-with-slurm]$ bash python.sh
/home/mtrappet/RaySlurm/use-ray-with-slurm/sbatch_template.sh
Path(__file__): /gpfs/home/mtrappet/RaySlurm/use-ray-with-slurm/launch.py
Start to submit job!
Job submitted! Script file is at: <test_0412-1452.sh>. Log file is at: <test_0412-1452.log>
(data-science) [mtrappet@login1 use-ray-with-slurm]$ Submitted batch job 2761882

(data-science) [mtrappet@login1 use-ray-with-slurm]$ nice watch squeue -u mtrappet
(data-science) [mtrappet@login1 use-ray-with-slurm]$ ls
launch.py  sbatch_template.sh  test_0412-1452.sh  test_1129-1508.log  test.py
python.sh  test_0412-1452.log  test_1129-1507.sh  test_1129-1508.sh
(data-science) [mtrappet@login1 use-ray-with-slurm]$ nano test_0412-1452.log
(data-science) [mtrappet@login1 use-ray-with-slurm]$ nano test.py
(data-science) [mtrappet@login1 use-ray-with-slurm]$ nano test_0412-1452.log
(data-science) [mtrappet@login1 use-ray-with-slurm]$ nano test.py
(data-science) [mtrappet@login1 use-ray-with-slurm]$ bash python.sh
/home/mtrappet/RaySlurm/use-ray-with-slurm/sbatch_template.sh
Path(__file__): /gpfs/home/mtrappet/RaySlurm/use-ray-with-slurm/launch.py
Start to submit job!
Job submitted! Script file is at: <test_0412-1459.sh>. Log file is at: <test_0412-1459.log>
(data-science) [mtrappet@login1 use-ray-with-slurm]$ Submitted batch job 2762005

(data-science) [mtrappet@login1 use-ray-with-slurm]$ nice watch squeue -u mtrappet
(data-science) [mtrappet@login1 use-ray-with-slurm]$ nano test_0412-1459.log 
(data-science) [mtrappet@login1 use-ray-with-slurm]$ nano test.py
(data-science) [mtrappet@login1 use-ray-with-slurm]$ bash python.sh
/home/mtrappet/RaySlurm/use-ray-with-slurm/sbatch_template.sh
Path(__file__): /gpfs/home/mtrappet/RaySlurm/use-ray-with-slurm/launch.py
Start to submit job!
Job submitted! Script file is at: <test_0412-1506.sh>. Log file is at: <test_0412-1506.log>
(data-science) [mtrappet@login1 use-ray-with-slurm]$ Submitted batch job 2762067

(data-science) [mtrappet@login1 use-ray-with-slurm]$ nice watch squeue -u mtrappet
(data-science) [mtrappet@login1 use-ray-with-slurm]$ nano test_0412-1506.log
(data-science) [mtrappet@login1 use-ray-with-slurm]$ exit
logout
Connection to login.talapas.uoregon.edu closed.
(base) matthew@d141-109 ~ % ssh mtrappet@login.talapas.uoregon.edu
(mtrappet@login.talapas.uoregon.edu) Password: 
(mtrappet@login.talapas.uoregon.edu) Duo two-factor login for mtrappet

Enter a passcode or select one of the following options:

 1. Duo Push to XXX-XXX-3550
 2. Duo Push to iPad Pro - Matt (iOS)
 3. Phone call to XXX-XXX-3550
 4. SMS passcodes to XXX-XXX-3550

Passcode or option (1-4): 1
Success. Logging you in...
# Welcome to Talapas!
# Data on Talapas is NOT backed up. Data management is each users responsibility.
#
# Need support? Please visit the Talapas Knowledge Base:
# https://hpcrcf.atlassian.net/wiki/spaces/TW/overview
#
# You've connected to our *new* Talapas cluster.
# Please see the release notes for information on what's changed.
# https://hpcrcf.atlassian.net/l/cp/1AizPjHm
#
Last login: Thu Apr 11 14:02:14 2024 from 184.171.111.58
# Storage usage in GB as of Mon Apr 22 10:00:12 2024
Fileset          User             UsedByUser  UsedByAll      Quota  Use%
home             mtrappet                 22          -        250     9
tau              mtrappet                 43        568       2048    28
[mtrappet@login2 ~]$ sinfo
PARTITION      AVAIL  TIMELIMIT  NODES  STATE NODELIST
compute           up 1-00:00:00     12  drain n[0073-0084]
compute           up 1-00:00:00     21    mix n[0016,0029,0049-0053,0089,0097,0103,0180-0189,0192]
compute           up 1-00:00:00     11  alloc n[0085-0088,0098-0100,0102,0190-0191,0193]
compute           up 1-00:00:00     59   idle n[0013-0015,0017-0028,0030-0036,0054-0072,0090-0096,0101,0104-0108,0135-0136,0194-0196]
computelong       up 14-00:00:0     19    mix n[0016,0049-0053,0097,0103,0180-0189,0192]
computelong       up 14-00:00:0      6  alloc n[0098-0100,0102,0190-0191]
computelong       up 14-00:00:0     36   idle n[0013-0015,0017-0024,0054-0072,0101,0104-0108]
gpu               up 1-00:00:00      1   plnd n0154
gpu               up 1-00:00:00     15    mix n[0149-0150,0153,0155-0156,0158,0164-0172]
gpu               up 1-00:00:00      6   idle n[0151,0157,0159-0160,0162-0163]
gpulong           up 14-00:00:0      1   plnd n0154
gpulong           up 14-00:00:0     11    mix n[0153,0155-0156,0164-0169,0171-0172]
gpulong           up 14-00:00:0      3   idle n[0157,0162-0163]
interactive       up   12:00:00      1    mix n0209
interactive       up   12:00:00     18   idle n[0199,0210-0212,0302-0313,0398-0399]
interactivegpu    up    8:00:00      1    mix n0301
interactivegpu    up    8:00:00      1   idle n0161
memory            up 1-00:00:00      1   comp n0375
memory            up 1-00:00:00      7    mix n[0142,0144,0146,0148,0372-0373,0378]
memory            up 1-00:00:00      2  alloc n[0374,0376]
memory            up 1-00:00:00      6   idle n[0141,0143,0145,0147,0377,0379]
memorylong        up 14-00:00:0      6    mix n[0142,0144,0146,0148,0372,0378]
memorylong        up 14-00:00:0      2  alloc n[0374,0376]
preempt           up 7-00:00:00      1   plnd n0154
preempt           up 7-00:00:00      1   comp n0375
preempt           up 7-00:00:00     12  drain n[0073-0084]
preempt           up 7-00:00:00     80    mix n[0016,0029,0037-0038,0040,0042,0049-0053,0089,0097,0103,0142,0144,0146,0148-0150,0152-0153,0155-0156,0158,0164-0172,0180-0189,0192,0209,0221,0230,0235,0237,0251,0254-0259,0301,0317,0330,0332-0333,0349-0351,0364,0372-0373,0378,0380,0387-0394,0396,1000]
preempt           up 7-00:00:00     20  alloc n[0085-0088,0098-0100,0102,0173-0179,0191,0193,0374,0376,0386]
preempt           up 7-00:00:00    201   idle n[0013-0015,0017-0028,0030-0036,0039,0041,0043-0044,0054-0072,0090-0096,0101,0104-0108,0135-0136,0141,0143,0145,0147,0151,0157,0159-0163,0194-0196,0199,0201-0208,0210-0220,0222-0229,0231-0234,0236,0238-0250,0252-0253,0260-0265,0302-0316,0318-0329,0331,0334-0348,0352-0363,0365-0371,0377,0379,0381-0385,0395,0397-0399]
[mtrappet@login2 ~]$ ls
Desktop    Downloads  main.cc  ray_results  survival-grid-world  Templates
Documents  foo.ppk    Public   RaySlurm     tau                  testing
[mtrappet@login2 ~]$ source survival-grid-world/data-science/bin/activate
(data-science) [mtrappet@login2 ~]$ cd RaySlurm/
(data-science) [mtrappet@login2 RaySlurm]$ ls
examples           start-head.sh    submit-ray-cluster.sbatch  test.out
groupmeeting12_14  start-worker.sh  test.err                   use-ray-with-slurm
(data-science) [mtrappet@login2 RaySlurm]$ cd use-ray-with-slurm/
(data-science) [mtrappet@login2 use-ray-with-slurm]$ ls
launch.py           test_0412-1452.log  test_0412-1459.sh   test_1129-1507.sh   test.py
python.sh           test_0412-1452.sh   test_0412-1506.log  test_1129-1508.log
sbatch_template.sh  test_0412-1459.log  test_0412-1506.sh   test_1129-1508.sh
(data-science) [mtrappet@login2 use-ray-with-slurm]$ rm test_*
(data-science) [mtrappet@login2 use-ray-with-slurm]$ ls
launch.py  python.sh  sbatch_template.sh  test.py
(data-science) [mtrappet@login2 use-ray-with-slurm]$ nano python.sh
(data-science) [mtrappet@login2 use-ray-with-slurm]$ bash python.sh
/home/mtrappet/RaySlurm/use-ray-with-slurm/sbatch_template.sh
Path(__file__): /gpfs/home/mtrappet/RaySlurm/use-ray-with-slurm/launch.py
Start to submit job!
Job submitted! Script file is at: <test_0422-1007.sh>. Log file is at: <test_0422-1007.log>
(data-science) [mtrappet@login2 use-ray-with-slurm]$ Submitted batch job 3181347

(data-science) [mtrappet@login2 use-ray-with-slurm]$ nice watch squeue -u mtrappet
(data-science) [mtrappet@login2 use-ray-with-slurm]$ ls
launch.py  python.sh  sbatch_template.sh  test_0422-1007.log  test_0422-1007.sh  test.py
(data-science) [mtrappet@login2 use-ray-with-slurm]$ nano test_0422-1007.log
(data-science) [mtrappet@login2 use-ray-with-slurm]$ nano test.p
(data-science) [mtrappet@login2 use-ray-with-slurm]$ nano test.py 
(data-science) [mtrappet@login2 use-ray-with-slurm]$ bash python.sh
/home/mtrappet/RaySlurm/use-ray-with-slurm/sbatch_template.sh
Path(__file__): /gpfs/home/mtrappet/RaySlurm/use-ray-with-slurm/launch.py
Start to submit job!
Job submitted! Script file is at: <test_0422-1014.sh>. Log file is at: <test_0422-1014.log>
(data-science) [mtrappet@login2 use-ray-with-slurm]$ Submitted batch job 3181531

(data-science) [mtrappet@login2 use-ray-with-slurm]$ nice watch squeue -u mtrappet
(data-science) [mtrappet@login2 use-ray-with-slurm]$ ls
launch.py  sbatch_template.sh  test_0422-1007.sh   test_0422-1014.sh
python.sh  test_0422-1007.log  test_0422-1014.log  test.py
(data-science) [mtrappet@login2 use-ray-with-slurm]$ nano test_0422-1014.log
(data-science) [mtrappet@login2 use-ray-with-slurm]$ nano test.py
(data-science) [mtrappet@login2 use-ray-with-slurm]$ nano test_0422-1014.log
(data-science) [mtrappet@login2 use-ray-with-slurm]$ rm test_*
(data-science) [mtrappet@login2 use-ray-with-slurm]$ bash python.sh
/home/mtrappet/RaySlurm/use-ray-with-slurm/sbatch_template.sh
Path(__file__): /gpfs/home/mtrappet/RaySlurm/use-ray-with-slurm/launch.py
Start to submit job!
Job submitted! Script file is at: <test_0422-1024.sh>. Log file is at: <test_0422-1024.log>
(data-science) [mtrappet@login2 use-ray-with-slurm]$ Submitted batch job 3181767

(data-science) [mtrappet@login2 use-ray-with-slurm]$ nice watch squeue -u mtrappet
(data-science) [mtrappet@login2 use-ray-with-slurm]$ ls
launch.py  python.sh  sbatch_template.sh  test_0422-1024.log  test_0422-1024.sh  test.py
(data-science) [mtrappet@login2 use-ray-with-slurm]$ nano test_0422-1024.log 
(data-science) [mtrappet@login2 use-ray-with-slurm]$ nano test.py
(data-science) [mtrappet@login2 use-ray-with-slurm]$ bash python.sh
/home/mtrappet/RaySlurm/use-ray-with-slurm/sbatch_template.sh
Path(__file__): /gpfs/home/mtrappet/RaySlurm/use-ray-with-slurm/launch.py
Start to submit job!
Job submitted! Script file is at: <test_0422-1026.sh>. Log file is at: <test_0422-1026.log>
(data-science) [mtrappet@login2 use-ray-with-slurm]$ Submitted batch job 3181835

(data-science) [mtrappet@login2 use-ray-with-slurm]$ nice watch squeue -u mtrappet
(data-science) [mtrappet@login2 use-ray-with-slurm]$ ls
launch.py  sbatch_template.sh  test_0422-1024.sh   test_0422-1026.sh
python.sh  test_0422-1024.log  test_0422-1026.log  test.py
(data-science) [mtrappet@login2 use-ray-with-slurm]$ nano test_0422-1026.log 
(data-science) [mtrappet@login2 use-ray-with-slurm]$ rm test_*
(data-science) [mtrappet@login2 use-ray-with-slurm]$ ls
launch.py  python.sh  sbatch_template.sh  test.py
(data-science) [mtrappet@login2 use-ray-with-slurm]$ nano launch.py
(data-science) [mtrappet@login2 use-ray-with-slurm]$ nano sbatch_template.sh

  GNU nano 2.9.8                                                    sbatch_template.sh                                                               

#!/bin/bash

# THIS FILE IS GENERATED BY AUTOMATION SCRIPT! PLEASE REFER TO ORIGINAL SCRIPT!
# THIS FILE IS A TEMPLATE AND IT SHOULD NOT BE DEPLOYED TO PRODUCTION!

#SBATCH --partition={{PARTITION_NAME}}
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output={{JOB_NAME}}.log
{{GIVEN_NODE}}

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes={{NUM_NODES}}
#SBATCH --exclusive

### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task={{NUM_GPUS_PER_NODE}}

#SBATCH --time=1-00:00:00     ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --account=tau  ### Account used for job submission
#SBATCH --mail-user=$onebonsai.warrior@gmail.com
#SBATCH --mail-type=ALL


# Load modules or your own conda environment here
# module load pytorch/v1.4.0-gpu
# conda activate {{CONDA_ENV}}
{{LOAD_ENV}}

################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address) # making redis-address

if [[ $ip == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$ip"
  if [[ ${#ADDR[0]} > 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "We detect space in ip! You are using IPV6 address. We split the IPV4 address as $ip"
fi

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
# srun --nodes=1 --ntasks=1 -w $node_1 start-head.sh $ip $redis_password &
srun --nodes=1 --ntasks=1 -w $node_1 \
  ray start --head --node-ip-address=$ip --port=6379 --redis-password=$redis_password --block &
sleep 30

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= $worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i ray start --address $ip_head --redis-password=$redis_password --block &
  sleep 5
done

##############################################################################################

#### call your code below
{{COMMAND_PLACEHOLDER}} {{COMMAND_SUFFIX}}