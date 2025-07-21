# IC-Fusion

## Network Architecture

## Installation
```
conda create -n icfusion python=3.10  # create a virtual env
pip install -r requirements.txt     # install other needed packages
```

## Dataset Preparation

## Test
- The benchmark results of our models can be downloaded from [IC-Fusion](https://gisto365-my.sharepoint.com/:u:/g/personal/sm_hwang_gm_gist_ac_kr/EXiyV-SY3_lDjiy6KoDPLsEBvFgIKPpOMXW-jZ-tnkeMQA?e=WhINuN).

Run the following script to test the trained model:

```sh
tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint --test-only
```

## Performance

## TODO list
- [ ] Add instructions
- [x] Add test code
- [ ] Add checkpoint files
- [x] Add training code

## Citation

## Acknowledgement
Thanks to Yian Zhao et al for releasing their official implementation of the [RT-DETR](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html) paper. Our code is heavily borrowed from the implementation.
