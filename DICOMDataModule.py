
from collections import OrderedDict

import pytorch_lightning as pl
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from sklearn.model_selection import StratifiedKFold, KFold
from itertools import islice
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
import SimpleITK as sitk
import os
import shutil
import time
import pydicom as dcm
from tqdm import tqdm
import scipy.integrate as integrate

import logging
from pytorch_lightning.trainer.supporters import CombinedLoader

logging.basicConfig(level=logging.DEBUG)

import util.transforms3d as transforms3d 

class PhaseSubset(Subset):
    def __init__(self, dataset, indices, phase):
        super().__init__(dataset, indices)
        self.phase = phase

    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, idx):
        if "train" not in self.phase:
            transform = self.dataset.transform
            self.dataset.transform = None
            result = super().__getitem__(idx)
            self.dataset.transform = transform
            return result
        return super().__getitem__(idx)

class AbstractDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        super(AbstractDataModule, self).__init__()

    def prepare_data(self):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError

    def teardown(self, stage):
        raise NotImplementedError

def _set_additional_properties(self, **kwargs) -> None:
    for key, value in kwargs.items():
        setattr(self, key, value)

class DICOMDataModule(AbstractDataModule):
    class ClinicalDataset(Dataset):
        def __init__(self, clinical_data, label):
            super().__init__()
            self.clinical_data = clinical_data
            self.label = label
            if not isinstance(self.clinical_data, np.ndarray):
                self.clinical_data = self.clinical_data.to_numpy()
        def __getitem__(self, idx):
            return self.clinical_data[idx], self.label[idx]
        def __len__(self):
            return len(self.label)

    class DVHDataset(Dataset):
        def __init__(self, dose_files, label, dose_features=["mean_dose", "V_5", "V_10", "V_15", "V_20"], to_cumulative=True):
            super().__init__()
            self.dose_data = list(dose_files)
            self.label = label
            self.dose_features = dose_features
            self.minmax = lambda x : x / 2000.0
            self.to_cumulative = to_cumulative

        def _get_dose(self, index):
            if self.dose_data is not None:
                if isinstance(self.dose_data[index], str):
                    self.dose_data[index] = self._parse_dose_file(self.dose_data[index])
                    return self.dose_data[index]
                else:
                    return self.dose_data[index]
            else:
                return None

        def _get_dose_file_contents(self, file_path):
            contents = []
            with open(file_path, "r") as file:
                while True:
                    content = {}
                    lastKey = None
                    while True:
                        line = file.readline()
                        if line == "\n" or line == "": break

                        if ":" in line:
                            temp = ":".join([w.strip() for w in line.strip().split(":")]).split(":")
                            content[temp[0]] = ":".join(temp[1:]) if len(temp[1:]) > 0 else temp[1]
                            lastKey = temp[0]
                        else:
                            if len(content) == 0:
                                temp = " ".join(line.strip().split()).split()
                                keys = []
                                indices = ["]" in t for t in temp]
                                p = 0
                                for idx, i in enumerate(indices):
                                    if i:
                                        keys.append(" ".join(temp[p:idx+1]))
                                        p = idx+1
                                content = OrderedDict({k: [] for k in keys})
                            elif lastKey is None:
                                line =  " ".join(line.strip().split()).split()
                                for idx, key in enumerate(content.keys()):
                                    content[key].append(line[idx])
                            else:
                                content[lastKey] += " " + " ".join(line.strip().split())
                    contents.append(content)
                    if line == "": break
            return contents

        def _parse_dose_file(self, file_path):
            result = []
            contents = self._get_dose_file_contents(file_path)

            # find lung structure
            lung_content_idx = []
            both_lung_idx = None
            for idx, content in enumerate(contents):
                if content.get("Structure", None) is None:
                    continue
                if "Lung" in content["Structure"]:
                    lung_content_idx.append(idx)
                if "Both Lung" in content["Structure"]:
                    both_lung_idx = idx
            
            # find mean dose
            mean_dose = None
            volume = None
            left_idx = right_idx = None

            if both_lung_idx is not None:
                for k,v in contents[both_lung_idx].items():
                    if "Mean Dose" in k:
                        if "cGy" in k:
                            mean_dose = float(v)
                        else:
                            mean_dose = float(v) * float(contents[both_lung_idx]['Volume [cm³]']) / 100
                volume = float(contents[both_lung_idx]['Volume [cm³]'])
            else:
                logging.info(f"Lung Content Found: {len(lung_content_idx)}")
                for idx in lung_content_idx:
                    if ("Left" in contents[idx]["Structure"]) or ("Lt" in contents[idx]["Structure"]):
                        left_idx = idx
                        logging.debug(f"Left Lung Content Found: {contents[idx]['Structure']}")
                    elif ("Right" in contents[idx]["Structure"]) or ("Rt" in contents[idx]["Structure"]):
                        right_idx = idx
                        logging.debug(f"Right Lung Content Found: {contents[idx]['Structure']}")

                logging.info(f"Calculating Both Lung")
                left_volume = float(contents[left_idx]['Volume [cm³]'])
                right_volume = float(contents[right_idx]['Volume [cm³]'])
                
                left_mean_dose = right_mean_dose = None
                
                for k, v in contents[left_idx].items():
                    if "Mean Dose" in k:
                        if "cGy" in k:
                            left_mean_dose = float(v)
                        else:
                            left_mean_dose = float(v) * float(left_volume) / 100

                for k, v in contents[right_idx].items():
                    if "Mean Dose" in k:
                        if "cGy" in k:
                            right_mean_dose = float(v)
                        else:
                            right_mean_dose = float(v) * float(right_volume) / 100

                mean_dose = (left_volume * left_mean_dose + right_volume * right_mean_dose) / (left_volume + right_volume)
                logging.info(f"Mean Dose: {mean_dose}, Left Mean Dose: {left_mean_dose}, \
                                Right Mean Dose {right_mean_dose}, Left Volume: {left_volume}, Right Volume: {right_volume}, Total Volume {left_volume + right_volume}")

            result.append(mean_dose / 4000.0)

            if "Cumulative" in contents[0]["Type"]:
                for feature in self.dose_features:
                    if "V" in feature:
                        target_dose = float(feature.split("_")[-1])
                        if both_lung_idx is not None:
                            data = contents[both_lung_idx+1]
                            x = np.array(data["Dose [cGy]"], dtype=np.float32)
                            dx = x[1] - x[0]
                            y = np.array(data["Ratio of Total Structure Volume [%]"], dtype=np.float32)
                            target_dose_idx = (np.abs(x - target_dose * 100) < dx)
                            if target_dose_idx.sum() < 1:
                                result.append(0.0)
                                logging.debug("Calculating Feature - File: {}, Feature: {}, Dose: {}".format(file_path, feature, 0.0))
                            else:
                                target_dose_idx = target_dose_idx.argmax()
                                result.append(y[target_dose_idx] / 100)
                                logging.debug("Calculating Feature - File: {}, Feature: {}, Dose: {}".format(file_path, feature, y[target_dose_idx]/100.0))
                        else:
                            left_lung_data = contents[left_idx+1]
                            right_lung_data = contents[right_idx+1]
                            x = np.array(left_lung_data["Dose [cGy]"], dtype = np.float32)
                            dx = x[1] - x[0]
                            target_dose_idx = (np.abs(x - target_dose * 100) < dx)
                            if target_dose_idx.sum() < 1:
                                result.append(0.0)
                                logging.debug("Calculating Feature - File: {}, Feature: {}, Dose: {}".format(file_path, feature, 0.0))
                            else:
                                target_dose_idx = target_dose_idx.argmax()
                                ly = np.array(left_lung_data["Ratio of Total Structure Volume [%]"], dtype=np.float32)
                                ry = np.array(right_lung_data["Ratio of Total Structure Volume [%]"], dtype=np.float32)
                                feature_dose = (left_volume * ly[target_dose_idx] + right_volume * ry[target_dose_idx]) / (left_volume + right_volume)
                                logging.debug("Calculating Feature - File: {}, Feature: {}, Dose: {}".format(file_path, feature, feature_dose/100.0))
                                result.append(feature_dose / 100.0)
                    else:
                        continue
            elif "Differential" in contents[0]["Type"]:
# calculate differential
                for feature in self.dose_features:
                    if "V" in feature:
                        target_dose = float(feature.split("_")[-1])
                        if both_lung_idx is not None:
                            data = contents[both_lung_idx+1]
                            x = np.array(data["Dose [cGy]"], dtype = np.float32)
                            dx = x[1] - x[0]
                            target_dose_idx = (np.abs(x - target_dose * 100) < dx)
                            if target_dose_idx.sum() < 1:
                                result.append(0.0)
                                logging.debug("Calculating Feature - File: {}, Feature: {}, Dose: {}".format(file_path, feature, 0.0))
                            else:
                                y = np.array(data["dVolume / dDose [cm³ / cGy]"], dtype= np.float32)              
                                int_y = 1.0 - integrate.cumtrapz(y, x, initial = 0.0) / volume
                                target_dose_idx = target_dose_idx.argmax()
                                result.append(int_y[target_dose_idx])
                                logging.debug("Calculating Feature - File: {}, Feature: {}, Dose: {}".format(file_path, feature, int_y[target_dose_idx]))
                        else:
                            left_lung_data = contents[left_idx+1]
                            right_lung_data = contents[right_idx+1]
                            x = np.array(left_lung_data["Dose [cGy]"], dtype = np.float32)
                            dx = x[1] - x[0]
                            target_dose_idx = (np.abs(x - target_dose * 100) < dx)
                            if target_dose_idx.sum() < 1:
                                result.append(0.0)
                                logging.debug("Calculating Feature - File: {}, Feature: {}, Dose: {}".format(file_path, feature, 0.0))
                            else:
                                ly = np.array(left_lung_data["dVolume / dDose [cm³ / cGy]"], dtype= np.float32)              
                                ry = np.array(right_lung_data["dVolume / dDose [cm³ / cGy]"], dtype= np.float32)              
                                int_ly= 1.0 - integrate.cumtrapz(ly, x, initial = 0.0) / left_volume
                                int_ry= 1.0 - integrate.cumtrapz(ry, x, initial = 0.0) / right_volume
                                target_dose_idx = target_dose_idx.argmax()
                                feature_dose = (left_volume * ly[target_dose_idx] + right_volume * ry[target_dose_idx]) / (left_volume + right_volume)
                                logging.debug("Calculating Feature - File: {}, Feature: {}, Dose: {}".format(file_path, feature, feature_dose))
                                result.append(feature_dose)
                    else:
                        continue
            else:
                raise RuntimeError(f"Unknown data format {contents[0]['Type']}")

            return np.array(result)


        def __getitem__(self, index):
            dose = self._get_dose(index)
            label = self.label[index]

            #dose = self.minmax(dose)

            return dose, label
        
        def __len__(self):
            return len(self.dose_data)
        
    class DICOMDataset(Dataset):
        """
        Implementation for datasets using DICOM.

        Attributes
        ----------
        data
        label
        transform
        target_transform
        cache

        """

        def __init__(self, data, label, mask=True, output_spacing=[], pad_output=[], crop_output=[], padder_constant=-1000.0, transform=None, target_transform=None, cache=False, cache_dir="cache",operate=None, **kwargs):
            """

            Parameters
            ----------
            data    : Sequence[str]
            label   : Sequence[str]
            transform   : Callable[[Arraylike], torch.Tensor]
            target_transform    : Callable[[Arraylike], torch.Tensor]
            """
            super().__init__()
            self.data = data
            self.label = label
            self.transform = transform
            self.target_transform = target_transform
            self.cache = cache
            self.cache_dir = cache_dir
            self.operate = operate
            self.pad_output = pad_output
            self.padder_constant = padder_constant
            self.crop_output = crop_output
            self.output_spacing = output_spacing
            self.mask = mask
            print("mask: {}".format(mask))
            _set_additional_properties(self, **kwargs)
            self.mask_name="Both Lung"

        
        def _load_image(self, idx, image_dir: str, cache: bool=False, mask=True):
            logging.debug(f"Loading image {image_dir}")
            isCached = False
            try:
                with open(os.path.join(self.cache_dir, str(idx), "image.pickle"), "rb") as f:
                    image = pickle.load(f)
                    isCached = True
            except Exception as e:
                reader = sitk.ImageSeriesReader()
                files = reader.GetGDCMSeriesFileNames(image_dir)
                reader.SetFileNames(files)
                image = reader.Execute()

                if mask:
                    rs_files = [file for file in os.listdir(image_dir) if "RS" == file[:2]] # assuming RTStruct file starts with RS
                    if len(rs_files) < 1:
                        print("Mask not found for", idx, image)


                    import numpy as np
                    from numpy.linalg import inv
                    import pydicom as dcm
                    import cv2 as cv

                    spacing = image.GetSpacing()
                    inv_direction = inv(np.array(image.GetDirection()).reshape(3,3)) # assuming 3x3 cosine direction matrix
                    origin = image.GetOrigin()
                    size = image.GetSize()

                    maskArray = np.zeros(size[::-1]).astype(np.uint8)

                    rs = dcm.read_file(os.path.join(image_dir, rs_files[0]))

                    roi_number = str(0)
                    for ss_roi in rs.StructureSetROISequence:
                        if ss_roi.ROIName == self.mask_name:
                            roi_number = ss_roi.ROINumber

                    roi_contour = None
                    for contour_info in rs.ROIContourSequence:
                        if contour_info.ReferencedROINumber == roi_number:
                            try:
                                roi_contour = contour_info.ContourSequence
                            except:
                                break

                    polygons = {}
                    for contour in roi_contour:
                        points = np.array(contour.ContourData).reshape(-1, 3)
                        pixels = (points - origin) @ inv_direction / spacing
                        pixels = pixels.round().astype(int)
                        slice_number = pixels[0,2]
                        pixels = pixels[:, :2] # remove z-axis
                        polygons.setdefault(slice_number, []).append(pixels.reshape(-1, 1, 2))
                
                    for slice_number, slice_polygons in polygons.items():
                        cv.drawContours(maskArray[slice_number], slice_polygons, -1, (255), -1)

                    if self.fill_holes:
                        logging.debug(f"Filling Mask with Kernel size {self.kernel_size}")
                        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
                        maskArray = cv.morphologyEx(maskArray, cv.MORPH_CLOSE, kernel)
                
                    mask = sitk.GetImageFromArray(maskArray)
                    mask.CopyInformation(image)

                    image = sitk.Mask(image, mask, -1000, 0)

            return image, isCached

        def __getitem__(self, index):
            image = self.data[index]
            label = self.label[index]

            imageFile, isCached = self._load_image(index, image, cache=self.cache, mask=self.mask)
            
            if isCached is False:
                with open(os.path.join(self.cache_dir, str(index), "image.pickle"), "wb") as f:
                    imageArray = sitk.GetArrayFromImage(imageFile)
                    """
                    boundingBox = np.zeros(6, dtype=int)
                    print("Log: calculating bounding box, ", index)
                    for i in range(3):
                        pstate = False
                        state = False
                        _op = {True: max, False: min}
                        temp = []
                        for idx, imageArraySlice in enumerate(np.swapaxes(imageArray, 0, i)):
                            if imageArraySlice.max() != imageArraySlice.min():
                                state = True

                            if pstate != state:
                                temp.append(idx)

                            pstate = state
                            state = False
                        
                        boundingBox[2*i] = temp[0]
                        boundingBox[2*i+1] = temp[-1]
                            
                    print(index, boundingBox)
                    imageArray = imageArray[boundingBox[0]:boundingBox[1],boundingBox[2]:boundingBox[3],boundingBox[4]:boundingBox[5]]
                    """
                                
                    #oldImage = imageFile
                    #imageFile = sitk.GetImageFromArray(imageArray)
                    #imageFile.SetDirection(oldImage.GetDirection())
                    #imageFile.SetSpacing(oldImage.GetSpacing())
                    #imageFile.SetOrigin(oldImage.GetOrigin() + np.array(imageFile.GetDirection()).reshape(3,3) @ np.array([boundingBox[4], boundingBox[0], boundingBox[2]]) * imageFile.GetSpacing())


                    
                    original_size = imageFile.GetSize()
                    original_spacing = imageFile.GetSpacing()
                    output_size = [
                        int(np.round(original_size[i] * (original_spacing[i] / self.output_spacing[i]))) for i in range(3)
                    ]
                    padding_size_upper = [0,0,0]
                    padding_size_lower = [0,0,0]
                    crop_size_upper = [0,0,0]
                    crop_size_lower = [0,0,0]

                    resampler = sitk.ResampleImageFilter()
                    resampler.SetReferenceImage(imageFile)
                    resampler.SetInterpolator(sitk.sitkBSpline)
                    resampler.SetOutputSpacing(self.output_spacing)
                    resampler.SetSize(output_size)
                    image = resampler.Execute(imageFile)

                    padder = sitk.ConstantPadImageFilter()
                    cropper = sitk.CropImageFilter()

                    if output_size[0] < self.pad_output[0]:
                        padding_size_upper[0] = padding_size_lower[0] = int(np.floor((self.pad_output[0] - output_size[0])/2))
                        padding_size_upper[1] = padding_size_lower[1] = int(np.floor((self.pad_output[1] - output_size[1])/2))
                        if self.pad_output[0] > (output_size[0] + 2 * padding_size_upper[0]):
                            padding_size_upper[0] += 1
                            padding_size_upper[1] += 1
                    else:
                        crop_size_upper[0] = crop_size_lower[0] = int(np.floor((output_size[0] - self.crop_output[0]) / 2))
                        crop_size_upper[1] = crop_size_lower[1] = int(np.floor((output_size[1] - self.crop_output[1]) / 2))
                        if self.crop_output[0] < (output_size[0] - 2 * crop_size_upper[0]):
                            crop_size_upper[0] += 1
                            crop_size_upper[1] += 1

                    if output_size[2] < self.pad_output[2]:
                        padding_size_upper[2] = padding_size_lower[2] = int(np.floor((self.pad_output[2] - output_size[2])/2))
                        if self.pad_output[2] > (output_size[2] + 2 * padding_size_upper[2]):
                            padding_size_upper[2] += 1
                    else:
                        crop_size_upper[2] = crop_size_lower[2] = int(np.floor((output_size[2] - self.crop_output[2]) / 2))
                        if self.crop_output[2] < (output_size[2] - 2 * crop_size_upper[2]):
                            crop_size_upper[2] += 1

                    padder.SetPadLowerBound(padding_size_lower)
                    padder.SetPadUpperBound(padding_size_upper)
                    padder.SetConstant(self.padder_constant)
                    
                    cropper.SetLowerBoundaryCropSize(crop_size_lower)
                    cropper.SetUpperBoundaryCropSize(crop_size_upper)

                    image = padder.Execute(image)
                    image = cropper.Execute(image)


                    pickle.dump(image, f)
            else:
                image = imageFile

            imageArray = sitk.GetArrayFromImage(image) # DICOM is one channel
            if self.input_is_one_dimension:
                imageArray = np.expand_dims(imageArray, 0)
            else:
                imageArray = np.tile(imageArray, (3, 1, 1, 1))
         
            # temporary scaling
            eps = 1e-8

            # add windowing or other rescaling methods
            #imageArray = (imageArray - imageArray.min()) / (imageArray.max() - imageArray.min() + eps)
            #imageArray = np.clip(imageArray, eps, 1.0)
            #imageArray = torch.Tensor(imageArray)

#            ct_window_level = -600.0
#            ct_window_width = 1500.0
#            ct_min = -1000
#            ct_max = 3048
#            ct_upper = ct_window_level + ct_window_width / 2.0
#            ct_lower = ct_window_level - ct_window_width / 2.0
#            imageArray = (imageArray - ct_lower) / (ct_upper - ct_lower + eps)
#            imageArray = np.clip(imageArray, 0.0, 1.0)
#            imageArray = torch.Tensor(imageArray)
            imageArray = (imageArray - imageArray.min())/(imageArray.max() - imageArray.min() + 1e-7)
            imageArray = torch.Tensor(imageArray)

            if self.target_transform is not None:
                label = self.target_transform(label)

            if self.transform is not None:
                imageArray = self.transform(imageArray)
            
            return imageArray, label

        def __len__(self):
            return len(self.label)
            
    def __init__(self, cfg, dose_features):
        super().__init__(cfg)

        # for data preparation
        catalog = cfg.get("catalog")
        df = pd.read_csv(catalog.get("path"), catalog.get("sep", ","))
        self.dose_features = dose_features
        self.path = catalog.get("base_dir", "NAS01/") + df["path"]
        self.target = df["target"]
        self.clinical = df[[column for column in df.columns if column not in ["path", "target", "dose", "RP_ID"]]]
        self.dose = catalog.get("base_dir") + df["dose"] if any(["dose" in column for column in df.columns]) else None
        self.cache = cfg.get("cache")
        self.transform = cfg.get("transform")
        self.target_transform = cfg.get("target_transform")
        self.operate = cfg.get("operate")
        self.save_splits = cfg.get("save_splits")
        self.load_splits = cfg.get("load_splits")
        self.splits = cfg.get("splits")
        self.cache_dir = cfg.get("cache_dir")
        self.output_spacing = cfg["dataset"].get("output_spacing")
        self.pad_output = cfg["dataset"].get("pad_output")
        self.crop_output= cfg["dataset"].get("crop_output")
        self.padder_constant = cfg["dataset"].get("padder_constant")
        self.fill_holes = cfg["dataset"].get("fill_holes")
        self.kernel_size = cfg["dataset"].get("kernel_size")
        self.input_is_one_dimension = cfg["input_is_one_dimension"]
        self.mask = cfg["dataset"].get("mask", True)
        print("mask: {}".format(self.mask))

    def prepare_data(self):
        if self.transform is not None:
            print(self.transform)
            logging.debug(f"Building Transform: {self.transform}")
            self.transform = nn.Sequential(
                *[getattr(globals()[transform["module"]], transform["name"])(**transform["params"]) for transform in self.transform ]
            )
            logging.debug(f"Transform: {self.transform}")
        if self.target_transform is not None:
            self.target_transform = nn.Sequential(
                *[getattr(globals()[k], v["name"])(**v["params"]) for k, v in self.target_transform]
            )
            logging.debug(f"Target Transform: {self.target_transform}")
        print(self.transform, self.target_transform)
        self.dataset = DICOMDataModule.DICOMDataset(self.path, 
                                                    self.target, 
                                                    output_spacing=self.output_spacing,
                                                    pad_output=self.pad_output,
                                                    crop_output=self.crop_output,
                                                    padder_constant= self.padder_constant,
                                                    transform=self.transform,
                                                    target_transfrom=self.target_transform,
                                                    cache=self.cache,
                                                    operate=self.operate,
                                                    cache_dir=self.cache_dir,
                                                    fill_holes = self.fill_holes,
                                                    kernel_size = self.kernel_size, input_is_one_dimension = self.input_is_one_dimension, mask=self.mask)
        self.dose_dataset = None if self.dose is None else DICOMDataModule.DVHDataset(self.dose, self.target, dose_features=self.dose_features)
        self.clinical_dataset = None if self.clinical is None else DICOMDataModule.ClinicalDataset(self.clinical, self.target)
                                                        

    def setup(self, stage):
        if self.cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            for idx, image in enumerate(self.path):
                os.makedirs(f"{self.cache_dir}/{idx}", exist_ok=True)
        if self.load_splits:
            pass
        else:
            self.train_indices = []
            self.val_indices = []
            self.test_indices = []

            total = list(range(0, len(self.dataset.label)))
            if self.cfg["dataset"]["stratified"]:
                fold = StratifiedKFold(n_splits=self.cfg["dataset"]["folds"], shuffle=True, random_state=self.cfg["dataset"].get("random_seed"))
                inner_fold = StratifiedKFold(n_splits=self.cfg["dataset"]["folds"] - 1, shuffle=True, random_state=self.cfg["dataset"].get("random_seed"))
            else:
                fold = KFold(n_splits=self.cfg["dataset"]["folds"], shuffle=True, random_state=self.cfg["dataset"].get("random_seed"))
                inner_fold = KFold(n_splits=self.cfg["dataset"]["folds"] - 1, shuffle=True, random_state=self.cfg["dataset"].get("random_seed"))
            
            self.train_indices, self.test_indices = next(islice(fold.split(total, self.target), self.cfg["dataset"]["fold"], None))
            temp = next(islice(inner_fold.split(self.train_indices, self.target[self.train_indices]), self.cfg["dataset"]["inner_fold"], None))
            self.val_indices = self.train_indices[temp[1]]
            self.train_indices = self.train_indices[temp[0]]

            # Old
            #self.train_indices, self.test_indices = random_split(total,
            #    [int((1 - self.splits["test"]) * len(self.dataset.label)), int(self.splits["test"] * len(self.dataset.label))], generator=torch.Generator().manual_seed(42))
            #self.train_indices, self.val_indices = random_split(self.train_indices,
            #    [int(self.splits["train"] * len(self.dataset.label)), int(self.splits["val"] * len(self.dataset.label))], generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        image_dataloader = DataLoader(PhaseSubset(self.dataset, self.train_indices, "train"), 
                          batch_size = self.cfg["dataset"].get("batch_size", DICOMDataModule.defaults["batch_size"]),
                          num_workers=self.cfg["dataset"].get("num_workers"))

        dose_dataloader = None if self.dose_dataset is None else DataLoader(Subset(self.dose_dataset, self.train_indices),
                          batch_size = self.cfg["dataset"].get("batch_size", DICOMDataModule.defaults["batch_size"]),
                          num_workers=self.cfg["dataset"].get("num_workers"))

        clinical_dataloader = None if self.clinical_dataset is None else DataLoader(Subset(self.clinical_dataset, self.train_indices), 
                          batch_size = self.cfg["dataset"].get("batch_size", DICOMDataModule.defaults["batch_size"]),
                          num_workers=self.cfg["dataset"].get("num_workers"))

        return CombinedLoader({"image" : image_dataloader}) if dose_dataloader is None else CombinedLoader({"image" : image_dataloader, "dose": dose_dataloader, "clinical": clinical_dataloader})

    def val_dataloader(self):
        image_dataloader = DataLoader(PhaseSubset(self.dataset, self.val_indices, "val"), 
                          batch_size = self.cfg["dataset"].get("batch_size", DICOMDataModule.defaults["batch_size"]),
                          num_workers=self.cfg["dataset"].get("num_workers"))

        dose_dataloader = None if self.dose_dataset is None else DataLoader(Subset(self.dose_dataset, self.val_indices),
                          batch_size = self.cfg["dataset"].get("batch_size", DICOMDataModule.defaults["batch_size"]),
                          num_workers=self.cfg["dataset"].get("num_workers"))

        clinical_dataloader = None if self.clinical_dataset is None else DataLoader(Subset(self.clinical_dataset, self.val_indices), 
                          batch_size = self.cfg["dataset"].get("batch_size", DICOMDataModule.defaults["batch_size"]),
                          num_workers=self.cfg["dataset"].get("num_workers"))
                          
        return CombinedLoader({"image" : image_dataloader}) if dose_dataloader is None else CombinedLoader({"image" : image_dataloader, "dose": dose_dataloader, "clinical" : clinical_dataloader})


    def test_dataloader(self):
        image_dataloader = DataLoader(PhaseSubset(self.dataset, self.test_indices, "test"), 
                          batch_size = self.cfg["dataset"].get("batch_size", DICOMDataModule.defaults["batch_size"]),
                          num_workers=self.cfg["dataset"].get("num_workers"))

        dose_dataloader = None if self.dose_dataset is None else DataLoader(Subset(self.dose_dataset, self.test_indices),
                          batch_size = self.cfg["dataset"].get("batch_size", DICOMDataModule.defaults["batch_size"]),
                          num_workers=self.cfg["dataset"].get("num_workers"))

        clinical_dataloader = None if self.clinical_dataset is None else DataLoader(Subset(self.clinical_dataset, self.test_indices), 
                          batch_size = self.cfg["dataset"].get("batch_size", DICOMDataModule.defaults["batch_size"]),
                          num_workers=self.cfg["dataset"].get("num_workers"))
                          
        return CombinedLoader({"image" : image_dataloader}) if dose_dataloader is None else CombinedLoader({"image" : image_dataloader, "dose": dose_dataloader, "clinical" : clinical_dataloader})


    def teardown(self, stage):
        #if self.save_config:
        #    pass
        #else:
        #    shutil.rmtree(self.cache_dir, ignore_errors=False, onerror=None)
        pass

    defaults = {
        "batch_size": 1
    }

if __name__ == "__main__":

    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
    cfg = {"catalog" : {
        "base_dir" : "/home/yando/NAS01/Databank/1_Radiology/in_house/Lung/",
        "path" : "/home/yando/Workspace/RESNET503D_SRP/catalog/rp_catalog_with_dose_and_clinical_new.csv",
        "sep" : ","
    },
    "cache": True,
    "cache_dir": "/home/yando/Workspace/RESNET503D_SRP/outputs/no_mask",
    "dataset": {
        "folds": 5,
        "stratified": True,
        "random_seed": 42,
        "fold": 0,
        "inner_fold": 0,
        "batch_size" : 1,
        "output_spacing" : [2.0, 2.0, 2.5],
        "pad_output" : [224, 224, 200],
        "padder_constant": -1000.0,
        "crop_output" : [224,224,200],
        "num_workers" : 1,
        "fill_holes" : True,
        "kernel_size" : 7,
        "mask" : False
    },
    "input_is_one_dimension" : False,
    "transform" : [
        {
            "name" : "RandomAffine3D",
            "module" : "transforms3d",
            "params" : {
                "yz_plane_args" : {
                    "degrees" : 30.0,
                    "interpolation" : "BILINEAR"
                },
                "zx_plane_args" : {
                    "degrees" : 30.0,
                    "interpolation" : "BILINEAR"
                },
                "xy_plane_args" : {
                    "degrees" : 30.0,
                    "interpolation" : "BILINEAR"
                }
            }
        }
    ]
    }



    for i in range(1):
        cfg["dataset"]["inner_fold"] = i
        cfg["transform"] = None
        dataModule = DICOMDataModule(cfg, dose_features = ["mean_dose", "V_5", "V_20"])
        dataModule.prepare_data()
        dataModule.setup(None)
        train_dataloader = dataModule.train_dataloader()
        val_dataloader = dataModule.val_dataloader()
        test_dataloader = dataModule.test_dataloader()

        for batch in tqdm(train_dataloader):
            image = batch["image"]
            dose = batch["dose"]
            clinical = batch["clinical"]
            print(dose, clinical, sep="\n")
        for batch in tqdm(val_dataloader):
            image = batch["image"]
            dose = batch["dose"]
            print(dose, clinical, sep="\n")
        for batch in tqdm(test_dataloader):
            image = batch["image"]
            dose = batch["dose"]
            print(dose, clinical, sep="\n")
            #print("saving plots...")
            #for idx, X_img in enumerate(X[0]):
            #    fig, ax = plt.subplots()
            #    ax.matshow(X_img, vmin=0.0, vmax=1.0)
            #    plt.savefig("temp/" + str(patient_id) + "/" + str(idx) + ".png")
            #    plt.close(fig)
