### Build the container

Make sure you are on a roihu-gpu login node. Then, make the necessary changes to the path to the `requirements.txt` file in the container definition file (`container.def`) and then run the following command to build the container:

```bash
cd /path/to/MultiWebOrganizerPlus/environments/roihu
apptainer build --fakeroot --bind="$TMPDIR:/tmp" LABEL_PIPELINE.sif container.def
```

This will create a Singularity image file named `LABEL_PIPELINE.sif` in the current directory. This can take up to 1 hour, so be patient.


### Simpler approach

Instead of building the container, you can just use the `python-vllm/0.19.1` module on the roihu-gpu nodes. This module provides a compatible Python environment with the necessary packages installed, and it is much faster to set up. You can load the module with:

```bash
module purge
module load python-vllm/0.19.1
```
