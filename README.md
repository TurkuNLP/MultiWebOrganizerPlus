# hpc_repo_template
Repository to be used as a template for work on different HPC environments, especially CSC's LUMI and Roihu.

First, run

```bash
python initialise_repo.py
```

If you want to automatically remove the initialisation script after the directories have been created, run

```bash
python initialise_repo.py --delete
```

This will create all the folders and subfolders for the template. The aim is to encourage maintaining an organised structure that is easy to use across different HPCs. Of course, it can be modified freely to suit your needs. 

Note that the default license is Apache 2.0. Remember to remove it if needed.

The template follows this logic:
- All code shared between different HPCs should reside in `scripts/` or `notebooks/`
- All HPC specific stuff, such as SLURM launch scripts should stay in `hpc/`.
- Enviroments, such as `venv`s and containers should stay in `environments`
- `data/`, `results/`, `cache/`, `logs/` and `models/` are in .gitignore and thus won't be synced.

### Repo structure

root/
│
├── src/
├── data/
├── models/
│
├── results/
│   └── figures/
│
├── scripts/
│   ├── data_processing/
│   ├── data_analysis/
│   ├── model_training/
│   └── model_evaluation/
│
├── configs/
│   ├── example.yaml
│   └── experiments/
│
├── hpc/
│   ├── lumi/
│   │   └── example.sh
│   └── roihu/
│       └── example.sh
│
├── environments/
│   ├── requirements.txt
│   ├── lumi/
│   │   └── container.def
│   └── roihu/
│       └── container.def
│
├── .gitignore
└── README.md
