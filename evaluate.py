from tqdm import tqdm
import numpy as np
import hydra
import torch
import json
import os


def generate_gt(split, lang_input_path, scene_root_path):
    gt_dict = {}
    scene_ids = {}
    with open(lang_input_path, "r") as f:
        raw_data = json.load(f)
    for query in tqdm(raw_data, desc="Initializing ground truths"):
        scene_id = query["scene_id"]
        scene_ids[scene_id] = True
        object_id = int(query["object_id"])
        scene_data = torch.load(os.path.join(scene_root_path, split, f"{scene_id}.pth"))
        if "object_ids" not in query:
            # for ScanRefer and Nr3D
            object_ids = [int(query["object_id"])]
        corners = scene_data["aabb_corner_xyz"][np.in1d(scene_data["aabb_obj_ids"], np.array(object_ids))]
        aabb_min_max_bound = np.stack((corners.min(1), corners.max(1)), axis=1)
        gt_dict[(scene_id, object_id, int(query["ann_id"]))] = {
            "aabb_bound": aabb_min_max_bound,
            "eval_type": query["eval_type"]
        }
    scene_ids = list(scene_ids.keys())
    return gt_dict, scene_ids


def parse_prediction(scene_ids, pred_file_root_path):
    pred_dict = {}
    for scene_id in scene_ids:
        with open(os.path.join(pred_file_root_path, f"{scene_id}.json"), "r") as f:
            scene_predictions = json.load(f)
        for scene_prediction in scene_predictions:
            corners = np.array(scene_prediction["aabb"], dtype=np.float32)
            aabb_min_max_bound = np.stack((corners.min(1), corners.max(1)), axis=1)
            pred_dict[(scene_id, int(scene_prediction["object_id"]), int(scene_prediction["ann_id"]))] = {
                "aabb_bound": aabb_min_max_bound
            }
    return pred_dict


@hydra.main(version_base=None, config_path="config", config_name="global_config")
def main(cfg):
    split = cfg.data.evaluation.split

    # prepare gt
    lang_input_path = getattr(cfg.data.lang_metadata, f"{split}_language_data")
    gt_data, scene_ids = generate_gt(split, lang_input_path, cfg.data.scene_dataset_path)

    # prepare predictions
    pred_data = parse_prediction(scene_ids, cfg.pred_path)

    evaluator = hydra.utils.instantiate(cfg.data.evaluator, verbose=True)
    evaluator.set_ground_truths(gt_data)

    _, _ = evaluator.evaluate(pred_data)


if __name__ == '__main__':
    main()
