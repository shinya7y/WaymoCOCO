# WaymoCOCO

This converter converts the Waymo Open Dataset to COCO format.
Current implementation supports to extract the 2D information of the dataset v1.2 (the version used in the Waymo Open Dataset challenge at CVPR 2020).

## Installation

Requirements

* Linux
* Python 3.6+
* TensorFlow 1.15.0, 2.0.0, 2.1.0

Example using conda

``` bash
conda create -n waymococo python=3.7
conda activate waymococo
pip install tensorflow==2.1.0
git clone https://github.com/shinya7y/WaymoCOCO.git
cd WaymoCOCO
```

## Download

1. Access https://waymo.com/open/download/ .
2. Fill in the form and check the license for user registration.
3. Download the Training, Validation, and Test sets to `${HOME}/data/waymotfrecord/training/`, `${HOME}/data/waymotfrecord/validation/`, and `${HOME}/data/waymotfrecord/testing/`, respectively.

You can download them quickly by the commands below. [Installing Google Cloud SDK](https://cloud.google.com/sdk/docs) may be needed in advance.

``` bash
gcloud auth login
# follow messages for authentication

gsutil -m cp -r gs://waymo_open_dataset_v_1_2_0_individual_files/training/ ${HOME}/data/waymotfrecord/
gsutil -m cp -r gs://waymo_open_dataset_v_1_2_0_individual_files/validation/ ${HOME}/data/waymotfrecord/
gsutil -m cp -r gs://waymo_open_dataset_v_1_2_0_individual_files/testing/ ${HOME}/data/waymotfrecord/
```

## Usage

### WaymoCOCO f0 (frame 0)

The Waymo Open Dataset is large, but for many cases, it's too large.
Using its subsets is useful when you would like to:
* do much trial and error before full training.
* evaluate the robustness of your method on the second dataset other than COCO.

This converter supports to extract 1/10 size dataset based on the ones place of frame index (e.g., frames 0, 10, 20, ..., 190).

``` bash
# convert val
python convert_waymo_to_coco.py \
    --tfrecord_dir ${HOME}/data/waymotfrecord/validation/ \
    --work_dir ${HOME}/data/waymococo_f0/ \
    --image_dirname val2020 \
    --image_filename_prefix val \
    --label_filename instances_val2020.json \
    --add_waymo_info \
    --frame_index_ones_place 0
# convert train
python convert_waymo_to_coco.py \
    --tfrecord_dir ${HOME}/data/waymotfrecord/training/ \
    --work_dir ${HOME}/data/waymococo_f0/ \
    --image_dirname train2020 \
    --image_filename_prefix train \
    --label_filename instances_train2020.json \
    --add_waymo_info \
    --frame_index_ones_place 0
# convert test
python convert_waymo_to_coco.py \
    --tfrecord_dir ${HOME}/data/waymotfrecord/testing/ \
    --work_dir ${HOME}/data/waymococo_f0/ \
    --image_dirname test2020 \
    --image_filename_prefix test \
    --label_filename image_info_test2020.json \
    --add_waymo_info \
    --frame_index_ones_place 0
```

### WaymoCOCO full

Full conversion is also available. Please note that a machine with 208-416 GB of CPU memory is needed for full training in the case of MMDetection v2.0.

``` bash
# convert val
python convert_waymo_to_coco.py \
    --tfrecord_dir ${HOME}/data/waymotfrecord/validation/ \
    --work_dir ${HOME}/data/waymococo_full/ \
    --image_dirname val2020 \
    --image_filename_prefix val \
    --label_filename instances_val2020.json \
    --add_waymo_info
# convert train
python convert_waymo_to_coco.py \
    --tfrecord_dir ${HOME}/data/waymotfrecord/training/ \
    --work_dir ${HOME}/data/waymococo_full/ \
    --image_dirname train2020 \
    --image_filename_prefix train \
    --label_filename instances_train2020.json \
    --add_waymo_info
# convert test
python convert_waymo_to_coco.py \
    --tfrecord_dir ${HOME}/data/waymotfrecord/testing/ \
    --work_dir ${HOME}/data/waymococo_full/ \
    --image_dirname test2020 \
    --image_filename_prefix test \
    --label_filename image_info_test2020.json \
    --add_waymo_info
```

### Other options

Please see [convert_waymo_to_coco.py](convert_waymo_to_coco.py).

## Acknowledgements

The files in waymo_open_dataset directory are borrowed from [the official code](https://github.com/waymo-research/waymo-open-dataset/) to mitigate dependency.
The official code and [Waymo-Dataset-Tool](https://github.com/RalphMao/Waymo-Dataset-Tool) (converter for KITTI format) were referred to write this converter.
