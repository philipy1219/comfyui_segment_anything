import os
import sys
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

import copy
import torch
import numpy as np
from PIL import Image
import logging
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import folder_paths
import comfy.model_management
from sam_hq.predictor import SamPredictorHQ
from sam_hq.build_sam_hq import sam_model_registry
from sam_hq.automatic import SamAutomaticMaskGeneratorHQ
from local_groundingdino.datasets import transforms as T
from local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from local_groundingdino.models import build_model as local_groundingdino_build_model
import glob
import folder_paths
import cv2

logger = logging.getLogger('comfyui_segment_anything')

sam_model_dir_name = "sams"
sam_model_list = {
    "sam_vit_h (2.56GB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    },
    "sam_vit_l (1.25GB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    },
    "sam_vit_b (375MB)": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    },
    "sam_hq_vit_h (2.57GB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"
    },
    "sam_hq_vit_l (1.25GB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth"
    },
    "sam_hq_vit_b (379MB)": {
        "model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth"
    },
    "mobile_sam(39MB)": {
        "model_url": "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt"
    }
}

groundingdino_model_dir_name = "grounding-dino"
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
    },
}

def get_bert_base_uncased_model_path():
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        print('grounding-dino is using models/bert-base-uncased')
        return comfy_bert_model_base
    return 'bert-base-uncased'

def list_files(dirpath, extensions=[]):
    return [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f)) and f.split('.')[-1] in extensions]


def list_sam_model():
    return list(sam_model_list.keys())


def load_sam_model(model_name):
    sam_checkpoint_path = get_local_filepath(
        sam_model_list[model_name]["model_url"], sam_model_dir_name)
    model_file_name = os.path.basename(sam_checkpoint_path)
    model_type = model_file_name.split('.')[0]
    if 'hq' not in model_type and 'mobile' not in model_type:
        model_type = '_'.join(model_type.split('_')[:-1])
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam_device = comfy.model_management.get_torch_device()
    sam.to(device=sam_device)
    sam.eval()
    sam.model_name = model_file_name
    return sam


def get_local_filepath(url, dirname, local_file_name=None):
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)

    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination:
        logger.warn(f'using extra model: {destination}')
        return destination

    folder = os.path.join(folder_paths.models_dir, dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        logger.warn(f'downloading {url} to {destination}')
        download_url_to_file(url, destination)
    return destination


def load_groundingdino_model(model_name):
    dino_model_args = local_groundingdino_SLConfig.fromfile(
        get_local_filepath(
            groundingdino_model_list[model_name]["config_url"],
            groundingdino_model_dir_name
        ),
    )

    if dino_model_args.text_encoder_type == 'bert-base-uncased':
        dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()
    
    dino = local_groundingdino_build_model(dino_model_args)
    checkpoint = torch.load(
        get_local_filepath(
            groundingdino_model_list[model_name]["model_url"],
            groundingdino_model_dir_name,
        ),
    )
    dino.load_state_dict(local_groundingdino_clean_state_dict(
        checkpoint['model']), strict=False)
    device = comfy.model_management.get_torch_device()
    dino.to(device=device)
    dino.eval()
    return dino


def list_groundingdino_model():
    return list(groundingdino_model_list.keys())


def groundingdino_predict(
    dino_model,
    image,
    prompt,
    threshold
):
    def load_dino_image(image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    def get_grounding_output(model, image, caption, box_threshold):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        return boxes_filt.cpu()

    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt = get_grounding_output(
        dino_model, dino_image, prompt, threshold
    )
    H, W = image.size[1], image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt


def create_pil_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        output_masks.append(Image.fromarray(np.any(mask, axis=0)))
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_images.append(Image.fromarray(image_np_copy))
    return output_images, output_masks


def create_tensor_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_image, output_mask = split_image_mask(
            Image.fromarray(image_np_copy))
        output_masks.append(output_mask)
        output_images.append(output_image)
    return (output_images, output_masks)


def split_image_mask(image):
    image_rgb = image.convert("RGB")
    image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_rgb)[None,]
    if 'A' in image.getbands():
        mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image_rgb, mask)


def sam_segment(
    sam_model,
    image,
    boxes
):
    if boxes.shape[0] == 0:
        return None
    sam_is_hq = False
    # TODO: more elegant
    if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name:
        sam_is_hq = True
    predictor = SamPredictorHQ(sam_model, sam_is_hq)
    image_np = np.array(image)
    image_np_rgb = image_np[..., :3]
    predictor.set_image(image_np_rgb)
    transformed_boxes = predictor.transform.apply_boxes_torch(
        boxes, image_np.shape[:2])
    sam_device = comfy.model_management.get_torch_device()
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(sam_device),
        multimask_output=False)
    masks = masks.permute(1, 0, 2, 3).cpu().numpy()
    return create_tensor_output(image_np, masks, boxes)

def create_seg_color_image(
        canvas_image: np.ndarray,
        sam_masks: list,
        ) -> np.ndarray:
    """Create segmentation color image.

    Args:
        input_image (Union[np.ndarray, Image.Image]): input image
        sam_masks (List[Dict[str, Any]]): SAM masks

    Returns:
        np.ndarray: segmentation color image
    """

    for idx, mask in enumerate(sam_masks):
        seg_mask = np.expand_dims(mask.astype(np.uint8), axis=-1)
        canvas_mask = np.logical_not(canvas_image.astype(bool)).astype(np.uint8)
        seg_color = np.array([idx+1], dtype=np.uint8) * seg_mask * canvas_mask
        canvas_image = canvas_image + seg_color

    return canvas_image

class SAMModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_sam_model(), ),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("SAM_MODEL", )

    def main(self, model_name):
        sam_model = load_sam_model(model_name)
        return (sam_model, )


class GroundingDinoModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_groundingdino_model(), ),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("GROUNDING_DINO_MODEL", )

    def main(self, model_name):
        dino_model = load_groundingdino_model(model_name)
        return (dino_model, )


class GroundingDinoSAMSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ('SAM_MODEL', {}),
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {}),
                "threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0,
                    "max": 1.0,
                    "step": 0.01
                }),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, grounding_dino_model, sam_model, image, prompt, threshold):
        res_images = []
        res_masks = []
        for item in image:
            item = Image.fromarray(
                np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGBA')
            boxes = groundingdino_predict(
                grounding_dino_model,
                item,
                prompt,
                threshold
            )
            if boxes.shape[0] == 0:
                break
            (images, masks) = sam_segment(
                sam_model,
                item,
                boxes
            )
            res_images.extend(images)
            res_masks.extend(masks)
        if len(res_images) == 0:
            _, height, width, _ = image.size()
            empty_mask = torch.zeros((1, height, width), dtype=torch.uint8, device="cpu")
            return (empty_mask, empty_mask)
        return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))

class AutomaticSAMSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ('SAM_MODEL', {}),
                "image": ('IMAGE', {}),
                "seg_color_mask": ("BOOLEAN", {"default": True}),
                "minimum_pixels": ("INT", {"default":64}),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, sam_model, image, seg_color_mask, minimum_pixels):
        res_masks = []
        res_images = []
        sam_is_hq = False
        # TODO: more elegant
        if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name:
            sam_is_hq = True
        local_sam = SamAutomaticMaskGeneratorHQ(SamPredictorHQ(sam_model, sam_is_hq), pred_iou_thresh=0.86, stability_score_thresh=0.92, min_mask_region_area=64)
        for item in image:
            item = np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)
            anns = local_sam.generate(item)
            masks = np.array([ann["segmentation"] for ann in anns])
            if seg_color_mask:
                tmp_masks = []
                tmp_images = []
                masks = sorted(masks, key=lambda x: np.sum(x[0].astype(np.uint32)))
                shapes = masks[0].shape
                canvas_image = np.zeros((*shapes, 1), dtype=np.uint8)
                seg_image = create_seg_color_image(canvas_image, masks)
                pixels = seg_image.reshape(-1, seg_image.shape[-1])
                unique_colors = np.unique(pixels, axis=0)
                for i in range(len(unique_colors)):
                    tmp_mask = np.all(seg_image==unique_colors[i],axis=-1).astype(np.uint8)
                    if tmp_mask.sum() >= minimum_pixels * minimum_pixels:
                        background = np.zeros_like(item)
                        region = cv2.bitwise_and(item, item, mask=tmp_mask)
                        extracted_region = cv2.add(region, background)
                        extracted_region = np.array(extracted_region).astype(np.float32) / 255.0
                        extracted_region = torch.from_numpy(extracted_region)[None,]
                        tmp_images.append(extracted_region)
                        tmp_mask = np.array(tmp_mask).astype(np.float32)
                        tmp_mask = torch.from_numpy(tmp_mask)[None,]
                        tmp_masks.append(tmp_mask)
                res_masks.extend(tmp_masks)
                res_images.extend(tmp_images)
            else:
                res_masks.extend(masks)
                res_images.append(item)
        return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))

class AutomaticClipClassifySegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": ([], ),
                "clipseg_name": ([], ),
                "image": ('IMAGE', {}),
                "mask": ('MASK', {}),
                "scale_small": ('FLOAT', {"default":1.2}),
                "scale_huge": ('FLOAT', {"default":1.6})
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("TEXT")

    def main(self, clip_name, clipseg_name, image, mask, scale_small, scale_huge):
        import mmcv
        result_list = []
        for img, mas in zip(image, mask):
            item = np.clip(255. * img.cpu().numpy(), 0, 255).astype(np.uint8)
            bbox = self.get_mask_bbox(mas)
            patch_small = mmcv.imcrop(img, bbox, scale_ratio=scale_small)
            patch_huge = mmcv.imcrop(img, bbox, scale_ratio=scale_huge)
            mask_categories = self.clip_classification(patch_small, class_list, 3 if len(class_list)>3 else len(class_list), clip_processor, clip_model)
            class_ids_patch_huge = self.clipseg_segmentation(patch_huge, mask_categories, clipseg_processor, clipseg_model).argmax(0)
            top_1_patch_huge = torch.bincount(class_ids_patch_huge.flatten()).topk(1).indices
            top_1_mask_category = mask_categories[top_1_patch_huge.item()]
            result_list.append(top_1_mask_category)
        return result_list

    def clip_classification(self, image, class_list, top_k, clip_processor, clip_model):
        inputs = clip_processor(text=class_list, images=image, return_tensors="pt", padding=True).to(comfy.model_management.get_torch_device())
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        if top_k == 1:
            class_name = class_list[probs.argmax().item()]
            return class_name
        else:
            top_k_indices = probs.topk(top_k, dim=1).indices[0]
            top_k_class_names = [class_list[index] for index in top_k_indices]
            return top_k_class_names

    def clipseg_segmentation(self, image, class_list, clipseg_processor, clipseg_model):
        inputs = clipseg_processor(
            text=class_list, images=[image] * len(class_list),
            padding=True, return_tensors="pt").to(comfy.model_management.get_torch_device())
        # resize inputs['pixel_values'] to the longesr side of inputs['pixel_values']
        h, w = inputs['pixel_values'].shape[-2:]
        fixed_scale = (512, 512)
        inputs['pixel_values'] = F.interpolate(
            inputs['pixel_values'],
            size=fixed_scale,
            mode='bilinear',
            align_corners=False)
        outputs = clipseg_model(**inputs)
        logits = torch.nn.functional.interpolate(outputs.logits[None], size=(h, w), mode='bilinear', align_corners=False)[0]
        return logits

    def get_mask_bbox(self, mask):
        """
        获取给定mask的包围盒 (bounding box)。
        参数：
            mask (np.ndarray): 二值掩码，0表示背景，1表示前景。
        返回：
            tuple: 包围盒 (x_min, y_min, x_max, y_max)。
        """
        # 获取非零元素（前景像素）的索引
        rows = np.any(mask, axis=1)  # 哪些行有前景
        cols = np.any(mask, axis=0)  # 哪些列有前景
        # 找到行和列中第一个和最后一个非零元素的位置
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        # 包围盒的坐标
        return x_min, y_min, x_max, y_max

class InvertMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK",)

    def main(self, mask):
        out = 1.0 - mask
        return (out,)

class IsMaskEmptyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }
    RETURN_TYPES = ["NUMBER"]
    RETURN_NAMES = ["boolean_number"]

    FUNCTION = "main"
    CATEGORY = "segment_anything"

    def main(self, mask):
        return (torch.all(mask == 0).int().item(), )