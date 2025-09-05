# NEURD Neuroglancer State Generator

## Overview

**Problem:** When developing and iterating on auto-proofread neurons in NEURD, frequent reprocessing makes it laborious to inspect changes using standalone Python tools. Visualizing meshes and skeletons in isolation does not provide context within the EM volume or nearby neurites.

**Solution:** This module dynamically generates a Neuroglancer state—including EM imagery, segmentation, proofread meshes, and skeletons—for one or more segment IDs. It fetches proofread meshes and skeletons, uploads them to S3, creates the necessary `info` JSON, and then prints a sharable Neuroglancer link.


## Capabilities

* **Global voxel alignment:** States open directly in Neuroglancer with correct global coordinates for correct spatial context.
* **Dynamic state generation:** New states with EM data, segmentation, proofread meshes, and skeletons are created on-the-fly from queried data. If autoproofread changes are made, the neuroglancer state can be updated dynamically.
* **NEURD integration:** Pulls auto-proofread meshes and skeletons directly from NEURD DataJoint table.
* **Dataset defaults:** Preconfigured for the H01 dataset, with adjustable parameters for others.
* Sets **color assignments** for uploaded meshes and skeletons.
* **Flexible input:** Accepts either queried DataJoint rows or explicit segment ID lists.
* Uploads auto-proofread meshes and skeletons in precomputed format to **S3** with a generated `info`
* Generates and opens a **shareable Neuroglancer link** ready for inspection.

* **Dual usage:**

  * **Notebook** (`NEURDNeuroglancerGeneration.ipynb`) for interactive exploration.
  * **CLI script** (`generate_ng_state.py`) for quick link generation:

    ```bash
    python generate_ng_state.py
    ```

  * Both approaches will generate and open the Neuroglancer link in your browser.

NOTE: This implementation is configured to the **H01 dataset** (see `config.yaml` ). In theory, it can be parameterized for other datasets by adjusting configuration values. For now, users only need to provide **segment IDs**, since the dataset bindings default to H01.


## Environment Setup

NOTE: `connects-neuvue` was forked locally and modified [adding issues to original connects-neuvue repo to come]. Hence this repo includes the root-level `connects-neuvue` modules. It is recommended you fork the full directory here to run the `NEURDNeuroglancerGeneration` module. 

```bash
gh repo clone aplbrain/neurd-ngl-viewer
```
```bash
cd connects_neuvue/NEURDNeuroglancerGeneration
```

This project uses [`uv`](https://docs.astral.sh/uv/) for Python environment management.

### Initialize environment

```bash
uv init
```

### Synchronize dependencies

```bash
uv sync
```

### Prerequisites

* **Python:** 3.8+
* **AWS Credentials:** Ensure `~/.aws/credentials` or environment variables are configured for S3 access.
* **NGLui:** v4 (only the latest version is supported).

---

## Configuration

See [`config.yaml`](./config.yaml)  for default H01 dataset and S3 settings:

* **Dataset bindings** (`api_dataset`, `cave_client_name`, layer sources).
* **S3 parameters** (`bucket`, `s3_base_path`).
* **Viewer defaults** (dimensions, layer names).
* **Volume info parameters** used to build Neuroglancer metadata.

---

## Example Usage

### Notebook

Run `NEURDNeuroglancerGeneration.ipynb` to interactively:

* Query rows from NEURD DataJoint tables.
* Generate a Neuroglancer link per queried neuron.
* Inspect EM imagery, segmentation, and proofread structures in context.

### Command Line

Generate a Neuroglancer link directly:

```bash
python generate_ng_state.py
```

Outputs:

* Meshes uploaded to S3.
* Info JSON created.
* Neuroglancer link printed to terminal.
---