import os
from os.path import basename, dirname, join
import pandas as pd
from glob import glob
from tqdm import tqdm
import shutil
import SimpleITK as sitk
# orig_df = "data/nii_info.xlsx"
import subprocess


def single_process_remove_neck_slices(nii_paths, out_dir):
    for nii_path in tqdm(nii_paths):
        group_name = basename(dirname(nii_path))
        pname = basename(nii_path).split(".nii")[0]
        output_path = join(out_dir, group_name, pname + ".nii.gz")

        os.makedirs(dirname(output_path), exist_ok=True)

        # First command
        head_top = subprocess.check_output("robustfov -i {} | grep -v Final | head -n 1 | awk '{{print $5}}'".format(nii_path), shell=True).decode().strip()

        print(head_top)

        # Second command
        subprocess.run("fslmaths {} -roi 0 -1 0 -1 {} 510 0 1 {}".format(nii_path, head_top, output_path), shell=True)

        # remove neck slices (zero slices)
        itk_img = sitk.ReadImage(output_path)
        itk_img_arr = sitk.GetArrayFromImage(itk_img)
        new_arr = []
        for i, img_slice in enumerate(itk_img_arr):
            if (img_slice != 0).sum() > 0:
                new_arr.append(img_slice)
            # else:
            #     if i >= itk_img_arr.shape[0] // 2:
            #         new_arr.append(img_slice)
                    
        itk_new_arr = sitk.GetImageFromArray(new_arr)
        itk_new_arr.SetOrigin(itk_img.GetOrigin())
        itk_new_arr.SetDirection(itk_img.GetDirection())
        itk_new_arr.SetSpacing(itk_img.GetSpacing())
        sitk.WriteImage(itk_new_arr, output_path)

        break


# multi-process preprocess
from multiprocessing import Process
def multi_process_preprocess(subjects, out_dir):
    process_list = []
    cpu_count = 4
    batch = len(subjects) // cpu_count
    for i in range(cpu_count):
        if i == (cpu_count - 1):
            this_subjects = subjects[i * batch:]
        else:
            this_subjects = subjects[i*batch : (i+1)*batch]
        p = Process(target=single_process_remove_neck_slices, args=(this_subjects, out_dir))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    print("preprocess end!")


from argparse import ArgumentParser
if __name__ == "__main__":
    # parser = ArgumentParser(description="convert grouped dicom (mixed by nii files) into niigz files")
    # parser.add_argument("--in_dir", type=str)
    # parser.add_argument("--out_dir", type=str)
    # args = parser.parse_args()

    in_dir = "/data/dingsd/mr2ct/data/orig_dcm_data/preprocessed/mr"
    out_dir = "data/orig_dcm_data/preprocessed/mr_betneck"

    nii_paths = list(glob(join(in_dir, "*.nii.gz")))
    nii_paths = sorted(nii_paths)

    # multi_process_preprocess(nii_paths, out_dir)

    single_process_remove_neck_slices(nii_paths, out_dir)