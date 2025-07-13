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
from local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict, get_phrases_from_posmap
from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from local_groundingdino.models import build_model as local_groundingdino_build_model
import glob
import folder_paths
import cv2
import torchvision
import torchvision.transforms as TS
from ram.models import ram, ram_plus
from ram import inference_ram
import json

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

ram_model_dir_name = "ram"
ram_model_list = {
    "ram_plus_vits_l": {
        "model_url": "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth"
    },
    "ram_vits_l": {
        "model_url": "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth"
    }
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

def groundingdino_predict_with_text_threshold(
    dino_model,
    image,
    prompt,
    threshold,
    text_threshold
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
        
    def get_grounding_output(model, image, caption, box_threshold, text_threshold):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        scores = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            scores.append(logit.max().item())

        return boxes_filt, torch.Tensor(scores), pred_phrases
    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt, scores, pred_phrases = get_grounding_output(
        dino_model, dino_image, prompt, threshold, text_threshold
    )
    H, W = image.size[1], image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt, scores, pred_phrases 

def list_ram_model():
    return list(ram_model_list.keys())

def load_ram_model(model_name):
    ram_checkpoint = get_local_filepath(
        ram_model_list[model_name]["model_url"], ram_model_dir_name
    )
    if "plus" in model_name:
        ram_model = ram_plus(pretrained=ram_checkpoint, image_size=384, vit='swin_l')
    else:
        ram_model = ram(pretrained=ram_checkpoint, image_size=384, vit='swin_l')
    device = comfy.model_management.get_torch_device()
    ram_model.to(device=device)
    ram_model.eval()
    return ram_model

def ram_predict(
    ram_model,
    image,
):
    def load_ram_image(image_pil):
        raw_image = image_pil.resize((384, 384))
        transform = TS.Compose(
            [
                TS.Resize((384, 384)),
                TS.ToTensor(),
                TS.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        raw_image = transform(raw_image).unsqueeze(0)  # 3, h, w
        return raw_image
    def get_ram_output(model, image):
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        with torch.no_grad():
            outputs = inference_ram(image, model)
        tags=outputs[0].replace(' |', ',')
        tags_chinese=outputs[1].replace(' |', ',')
        return tags, tags_chinese
    ram_image = load_ram_image(image)
    tags, tags_chinese = get_ram_output(ram_model, ram_image)
    return tags, tags_chinese

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

class RAMModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_ram_model(), ),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("RAM_MODEL", )

    def main(self, model_name):
        ram_model = load_ram_model(model_name)
        return (ram_model, )

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
                "points_per_side": ("INT", {"default":32})
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, sam_model, image, seg_color_mask, minimum_pixels, points_per_side):
        res_masks = []
        res_images = []
        sam_is_hq = False
        # TODO: more elegant
        if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name:
            sam_is_hq = True
        local_sam = SamAutomaticMaskGeneratorHQ(SamPredictorHQ(sam_model, sam_is_hq), pred_iou_thresh=0.86, stability_score_thresh=0.92, min_mask_region_area=minimum_pixels, points_per_side=points_per_side)
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

class RAMSAMSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ('SAM_MODEL', {}),
                "dino_model": ('GROUNDING_DINO_MODEL', {}),
                "ram_model": ('RAM_MODEL', {}),
                "image": ('IMAGE', {}),
                "box_threshold": ("FLOAT", {
                    "default": 0.25,
                    "min": 0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "text_threshold": ("FLOAT", {
                    "default": 0.2,
                    "min": 0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "iou_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0,
                    "max": 1.0,
                    "step": 0.01
                })
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING", "STRING")
    RETURN_NAMES = ["image", "mask", "combined_mask", "bbox_info", "mask_labels"]

    def main(self, sam_model, dino_model, ram_model, image, box_threshold, text_threshold, iou_threshold):
        res_images = []
        res_masks = []
        bbox_info_list = []
        all_mask_labels = []  # 新增存储所有mask的标签
        # TODO: more elegant
        if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name:
            sam_is_hq = True
        local_sam = SamPredictorHQ(sam_model, sam_is_hq)
        device = comfy.model_management.get_torch_device()
        for item in image:
            item = np.clip(255. * item.cpu().numpy(), 0, 255).astype(np.uint8)
            image_pil = Image.fromarray(item)        
            tags, tags_chinese = ram_predict(ram_model, image_pil)
            boxes_filt, scores, pred_phrases = groundingdino_predict_with_text_threshold(
                dino_model, image_pil, tags, box_threshold, text_threshold
            )
            print(f"Before NMS: {boxes_filt.shape[0]} boxes")
            nms_idx = torchvision.ops.nms(boxes_filt, scores, iou_threshold).numpy().tolist()
            boxes_filt = boxes_filt[nms_idx]
            pred_phrases = [pred_phrases[idx] for idx in nms_idx]
            print(f"After NMS: {boxes_filt.shape[0]} boxes")
            image = np.array(image_pil)
            local_sam.set_image(image)
            transformed_boxes = local_sam.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
            masks, _, _ = local_sam.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(device),
                multimask_output = False,
            )
            tmp_images = []
            combined_mask = torch.zeros_like(masks[0][0])
            
            # 计算每个mask的面积并创建排序索引
            mask_areas = []
            for idx, mask in enumerate(masks):
                area = torch.sum(mask[0]).item()
                mask_areas.append((area, idx))
            
            # 按面积从大到小排序
            mask_areas.sort(key=lambda x: x[0], reverse=True)
            
            # 按照排序后的顺序处理masks
            for area, idx in mask_areas:
                mask = masks[idx]
                combined_mask = combined_mask | mask[0]
                mask_np = mask[0].cpu().numpy().astype(np.uint8) * 255
                background = np.zeros_like(item)
                region = cv2.bitwise_and(item, item, mask=mask_np)
                extracted_region = cv2.add(region, background)
                extracted_region = np.array(extracted_region).astype(np.float32) / 255.0
                extracted_region = torch.from_numpy(extracted_region)[None,]
                tmp_images.append(extracted_region)
                all_mask_labels.append(pred_phrases[idx])  # 保存当前mask对应的标签
                
            # 按照排序后的顺序添加masks
            sorted_masks = [masks[idx] for area, idx in mask_areas]
            res_masks.extend(sorted_masks)
            res_images.extend(tmp_images)
            
            # 按照排序后的顺序添加bbox信息
            sorted_boxes = [boxes_filt[idx].tolist() for area, idx in mask_areas]
            bbox_info_list.extend(sorted_boxes)
            
        bbox_json = json.dumps(bbox_info_list, ensure_ascii=False)
        labels_json = json.dumps(all_mask_labels, ensure_ascii=False)  # 将标签转换为JSON字符串
        
        return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0), combined_mask.unsqueeze(0), bbox_json, labels_json)

class CalculateMaskCenters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK", {}),
                "depth_image": ("IMAGE", {})
            }
        }
    
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mask_centers",)
    
    def calculate_depth(self, x_cord, y_cord, depth_npy):
        # 获取深度图维度，考虑可能有多个通道
        if len(depth_npy.shape) > 2:
            # 如果是多通道图像，使用第一个通道或平均值
            if depth_npy.shape[2] == 1:
                depth_npy = depth_npy[:, :, 0]
            else:
                # 使用所有通道的平均值
                depth_npy = np.mean(depth_npy, axis=2)
        
        h, w = depth_npy.shape
        x0, y0 = int(np.floor(x_cord)), int(np.floor(y_cord))
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        
        # 计算插值权重
        wx = x_cord - x0
        wy = y_cord - y0
        
        # 双线性插值
        top = depth_npy[y0, x0] * (1 - wx) + depth_npy[y0, x1] * wx
        bottom = depth_npy[y1, x0] * (1 - wx) + depth_npy[y1, x1] * wx
        
        return float(top * (1 - wy) + bottom * wy)
    
    def main(self, masks, depth_image): 
        import json
        # 转换深度图为numpy数组
        depth_np = depth_image[0].cpu().numpy()
        
        # 初始化结果列表
        mask_centers = []
        
        # 遍历每个mask
        for i in range(masks.shape[0]):
            mask = masks[i].cpu().numpy()
            
            # 找到mask中所有非零点的坐标
            y_coords, x_coords = np.where(mask > 0)
            
            if len(y_coords) > 0:
                # 计算mask的中心点
                center_y = np.mean(y_coords)
                center_x = np.mean(x_coords)
                
                # 计算中心点的深度值
                center_depth = self.calculate_depth(center_x, center_y, depth_np)
                
                # 将中心坐标添加到结果列表
                mask_centers.append((float(center_x)/depth_image.shape[1], float(center_y)/depth_image.shape[0], float(center_depth)))
        
        return (json.dumps(mask_centers,ensure_ascii=False),)

class MaskToRandomLatentNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",),
            }
        }

    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("LATENT",)

    def main(self, image, mask, vae):
        # 编码图像获取latent
        latent = vae.encode(image[:,:,:,:3])
        
        # 生成随机噪声,形状与latent相同
        noise = torch.randn_like(latent)
        
        # 调整mask形状以匹配latent
        mask_downsample = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(latent.shape[2], latent.shape[3]), mode='bilinear')
        mask_downsample = mask_downsample.squeeze(0)
        
        # 使用广播方式
        mask_condition = mask_downsample.unsqueeze(1)  # [batchsize, 1, w, h]
        result = torch.where(mask_condition > 0.5, noise, latent)
        
        # 返回latent字典格式
        return ({"samples": result},)


class ComputeSurfaceTiltAngleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),  # 多张mask
                "depth_image": ("IMAGE",),  # 单张深度图
            }
        }

    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("surface_angles",)
    
    def main(self, masks, depth_image):
        import json
        
        # 转换深度图为numpy数组
        depth_np = depth_image[0].cpu().numpy()
        # 获取深度图维度，考虑可能有多个通道
        if len(depth_np.shape) > 2:
            # 如果是多通道图像，使用第一个通道或平均值
            if depth_np.shape[2] == 1:
                depth_np = depth_np[:, :, 0]
            else:
                # 使用所有通道的平均值
                depth_np = np.mean(depth_np, axis=2)

        # 计算深度图的梯度
        # 使用Sobel算子计算x和y方向的梯度
        depth_dx = cv2.Sobel(depth_np, cv2.CV_32F, 1, 0, ksize=3)
        depth_dy = cv2.Sobel(depth_np, cv2.CV_32F, 0, 1, ksize=3)

        # 存储所有mask的倾斜角度和相机视角角度
        surface_angles = []
        
        # 遍历每个mask
        for i in range(masks.shape[0]):
            mask = masks[i].cpu().numpy()
            
            # 找到mask中所有非零点的坐标
            ys, xs = np.where(mask > 0)
            
            if len(ys) < 10:  # 确保有足够的点进行拟合
                surface_angles.append(-1)
                continue
                
            # 提取该区域的梯度
            mask_dx = depth_dx[ys, xs]
            mask_dy = depth_dy[ys, xs]
            
            # 计算平均梯度或者中值梯度（中值可能更鲁棒）
            avg_dx = np.median(mask_dx)
            avg_dy = np.median(mask_dy)
            
            # 根据梯度构建法向量
            # 在深度图中，梯度与法向量的关系：法向量 = (-dx, -dy, 1)
            normal = np.array([-avg_dx, -avg_dy, 1.0])
            
            # 归一化法向量
            normal_magnitude = np.linalg.norm(normal)
            if normal_magnitude > 0:
                normal = normal / normal_magnitude

            # 相机视角向量（假设是正视图，指向z轴正方向）
            camera_vector = np.array([0, 0, 1])
            
            # 计算法向量与相机视角的夹角
            cos_angle = np.dot(normal, camera_vector)
            angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)
            
            surface_angles.append(float(angle_deg))
        
        return (json.dumps(surface_angles, ensure_ascii=False),)

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