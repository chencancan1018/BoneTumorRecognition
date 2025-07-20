import argparse
import os
import SimpleITK as sitk
from predictor import SegPredictor, ClsPredictor
import tarfile
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Test for abdomen_seg_mask3d')
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--data_dir', default='/media/tx-deepocean/Data/Infervision/InferMatrix模型集成/北大人民骨肿瘤分割分类_datademo/data_demo/img/80521057', type=str)
    parser.add_argument('--save_dir', default='/media/tx-deepocean/Data/Infervision/InferMatrix模型集成/北大人民骨肿瘤分割分类_datademo/data_demo/output', type=str)
    parser.add_argument('--model', default='/media/tx-deepocean/Data/Infervision/InferMatrix模型集成/北大人民骨肿瘤分割分类/infer/model', type=str)
    args = parser.parse_args()
    return args

def read_series(img_path):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(img_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.ProcessObject_SetGlobalWarningDisplay(True)
    return image

def main(input_path, output_path, gpu, args):
    cur_tar = tarfile.open(args.model+'.tar')
    seg_pth = cur_tar.extractfile('model/seg220928_1-checkpoint0104.pth')
    cls_pth = cur_tar.extractfile('model/cls220701_4-checkpoint0034.pth')

    print(input_path)
    if input_path.endswith('.nii'):
        itk_img = sitk.ReadImage(input_path)
    else:
        itk_img = read_series(input_path)

    pid = os.path.split(input_path)[-1]
    if pid.endswith(".nii"):
        pid = pid.replace('.nii', '')
    elif pid.endswith(".nii.gz"):
        pid = pid.replace(".nii.gz", '')

    img_array = sitk.GetArrayFromImage(itk_img)
    os.makedirs(output_path, exist_ok=True)

    seg_predictor = SegPredictor(gpu=gpu, model_pth=seg_pth)
    mask = seg_predictor.predict(img_array)

    itk_mask = sitk.GetImageFromArray(mask)
    itk_mask.CopyInformation(itk_img)
    output_dir = os.path.join(output_path, pid+'-seg.nii.gz')
    sitk.WriteImage(itk_mask, output_dir)

    cls_predictor = ClsPredictor(gpu=gpu, model_pth=cls_pth)
    pred_prob = cls_predictor.predict(img_array, mask)
    thr = 0.9109
    y_pred_0 = pred_prob[0]
    y_pred_1 = pred_prob[1]
    if y_pred_1 >= thr:
        label = '恶性'
    else:
        label='良性'

    result = dict()
    result["data_path"] = [input_path]

    storage_path = dict()
    storage_path["bone_tumor"] = output_dir
    result["storage_path"] = storage_path

    console = []
    seg_console = dict()
    seg_console["name"] = "bone_tumor"
    seg_console["opacity"] = 1
    seg_console["contour"] = True
    children = list()

    info = dict()
    info["task_type"] = "bone_tumor"
    info["mask_label"] = 1
    info["color"] = 'FF0000'
    info["name"] = "骶骨"
    info["opacity"] = 1
    info["contour"] = True
    children.append(info)

    seg_console["children"] = children
    console.append(seg_console)

    result["console"] = console
    result["lesion"] = {}
    result['title'] = ['mask_label', '类型', '良性概率', '恶性概率', '阈值']
    result['data'] = [['{mask_label}', label, str(y_pred_0), str(y_pred_1), str(thr)]]
    print(result)
    json_str = json.dumps(result, ensure_ascii=False, indent=4)
    json_file = os.path.join(args.save_dir, "result.json")
    with open(json_file, "w") as fp:
        fp.write(json_str)

    return

if __name__ == '__main__':
    args = parse_args()
    main(
        input_path=args.data_dir,
        output_path=args.save_dir,
        gpu=args.gpu,
        args=args,
    )
