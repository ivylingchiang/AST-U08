u08@nelson-Lab:~/workspace$ hostname
lscpu
free -h
nvidia-smi
df -h
nelson-Lab
Architecture:                x86_64
  CPU op-mode(s):            32-bit, 64-bit
  Address sizes:             48 bits physical, 48 bits virtual
  Byte Order:                Little Endian
CPU(s):                      16
  On-line CPU(s) list:       0-15
Vendor ID:                   AuthenticAMD
  Model name:                AMD Ryzen 7 7700 8-Core Processor
    CPU family:              25
    Model:                   97
    Thread(s) per core:      2
    Core(s) per socket:      8
    Socket(s):               1
    Stepping:                2
    Frequency boost:         enabled
    CPU(s) scaling MHz:      80%
    CPU max MHz:             5392.8721
    CPU min MHz:             422.3340
    BogoMIPS:                7585.32
    Flags:                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdp
                             e1gb rdtscp lm constant_tsc rep_good amd_lbr_v2 nopl xtopology nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor 
                             ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse
                              3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpuid_fault cpb cat_l3 cdp_l3 hw_ps
                             tate ssbd mba perfmon_v2 ibrs ibpb stibp ibrs_enhanced vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq
                              rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc 
                             cqm_mbm_total cqm_mbm_local user_shstk avx512_bf16 clzero irperf xsaveerptr rdpru wbnoinvd cppc arat npt lbrv svm_lock nrip_save ts
                             c_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic vgif x2avic v_spec_ctrl vnmi avx512vbmi umip pku ospke av
                             x512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid overflow_recov succor smca fsrm flush_l1d amd_lbr_
                             pmc_freeze
Virtualization features:     
  Virtualization:            AMD-V
Caches (sum of all):         
  L1d:                       256 KiB (8 instances)
  L1i:                       256 KiB (8 instances)
  L2:                        8 MiB (8 instances)
  L3:                        32 MiB (1 instance)
NUMA:                        
  NUMA node(s):              1
  NUMA node0 CPU(s):         0-15
Vulnerabilities:             
  Gather data sampling:      Not affected
  Ghostwrite:                Not affected
  Indirect target selection: Not affected
  Itlb multihit:             Not affected
  L1tf:                      Not affected
  Mds:                       Not affected
  Meltdown:                  Not affected
  Mmio stale data:           Not affected
  Old microcode:             Not affected
  Reg file data sampling:    Not affected
  Retbleed:                  Not affected
  Spec rstack overflow:      Mitigation; Safe RET
  Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
  Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
  Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; STIBP always-on; PBRSB-eIBRS Not affected; BHI Not affected
  Srbds:                     Not affected
  Tsa:                       Mitigation; Clear CPU buffers
  Tsx async abort:           Not affected
  Vmscape:                   Mitigation; IBPB before exit to userspace
               total        used        free      shared  buff/cache   available
Mem:            30Gi       7.8Gi        13Gi       526Mi       9.8Gi        22Gi
Swap:          8.0Gi       7.7Gi       272Mi
Tue Jun  9 20:29:53 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060        Off |   00000000:01:00.0  On |                  N/A |
| 53%   51C    P2             49W /  170W |    2638MiB /  12288MiB |      4%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A         3231287      G   /usr/lib/xorg/Xorg                     1126MiB |
|    0   N/A  N/A         3231535      G   /usr/bin/gnome-shell                     79MiB |
|    0   N/A  N/A         3232064      G   ...exec/xdg-desktop-portal-gnome          2MiB |
|    0   N/A  N/A         3232321    C+G   sunshine                                544MiB |
|    0   N/A  N/A         3264011      G   ...g/discord/app-1.0.139/Discord        186MiB |
|    0   N/A  N/A         3728308      G   .../8417/usr/lib/firefox/firefox         56MiB |
|    0   N/A  N/A         3810845    C+G   ...rack-uuid=3190708988185955192        105MiB |
|    0   N/A  N/A         3812852      G   ...share/Steam/ubuntu12_32/steam          3MiB |
|    0   N/A  N/A         3813050      G   ./steamwebhelper                          5MiB |
|    0   N/A  N/A         3813086    C+G   ...am/ubuntu12_64/steamwebhelper          5MiB |
|    0   N/A  N/A         3817533      G   ...nap-store/1367/bin/snap-store        244MiB |
+-----------------------------------------------------------------------------------------+
Filesystem      Size  Used Avail Use% Mounted on
tmpfs           3.1G  4.5M  3.1G   1% /run
/dev/nvme0n1p5  768G  345G  385G  48% /
tmpfs            16G  877M   15G   6% /dev/shm
tmpfs           5.0M   16K  5.0M   1% /run/lock
efivarfs        128K   64K   60K  52% /sys/firmware/efi/efivars
/dev/nvme0n1p1   96M   38M   59M  40% /boot/efi
tmpfs           3.1G  7.3M  3.1G   1% /run/user/1000
tmpfs           3.1G  104K  3.1G   1% /run/user/1003
tmpfs           3.1G  104K  3.1G   1% /run/user/1004
tmpfs           3.1G  200K  3.1G   1% /run/user/1005
tmpfs           3.1G   88K  3.1G   1% /run/user/1006