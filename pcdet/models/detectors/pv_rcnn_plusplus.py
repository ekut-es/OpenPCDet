import torch

from .detector3d_template import Detector3DTemplate


class PVRCNNPlusPlus(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.backbone_3d(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)

        batch_dict = self.roi_head.proposal_layer(
            batch_dict, nms_config=self.roi_head.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )


        # -------------------------------------------------------------------------------------------------------------
        # insert gt boxed to region proposels
        # the rois, scores and labels are shifted to the right by the number of gt boxes
        # then the boxes with the lowest scores (now the ones on the left) are overwritten with the gt boxes, scores, labels ...
        # dimensions are kept the same
        if self.model_cfg.get('COOP', None) is not None and self.model_cfg.COOP.USE_COOP_PROPOSALS_IN_PFE:
            coop_boxes = batch_dict['coop_boxes']
            coop_names = batch_dict['coop_ids']
            coop_scores = batch_dict['coop_scores']

            for batch_idx in range(batch_dict['batch_size']):
                num_gt = torch.count_nonzero(coop_names[batch_idx]).item()
                batch_dict['rois'][batch_idx] = torch.roll(batch_dict['rois'][batch_idx], num_gt, dims=0)
                batch_dict['rois'][batch_idx, :num_gt] = coop_boxes[batch_idx]

                batch_dict['roi_labels'][batch_idx] = torch.roll(batch_dict['roi_labels'][batch_idx], num_gt, dims=0)
                batch_dict['roi_labels'][batch_idx, :num_gt] = coop_names[batch_idx]

                batch_dict['roi_scores'][batch_idx] = torch.roll(batch_dict['roi_scores'][batch_idx], num_gt, dims=0)
                batch_dict['roi_scores'][batch_idx, :num_gt] = coop_scores[batch_idx]

        # -------------------------------------------------------------------------------------------------------------



        if self.training:

            # # todo try this, might speed up training
            # # # coop target assignment
            # coop_dict = {'rois': batch_dict['coop_boxes'], 'roi_labels': batch_dict['coop_ids'].long(),
            #              'roi_scores': batch_dict['coop_scores'], 'batch_size': batch_dict['batch_size'],
            #              'gt_boxes': batch_dict['gt_boxes']}
            # coop_target_dict = self.roi_head.assign_targets(coop_dict)
            # batch_dict['coop_rois'] = coop_target_dict['rois']
            # batch_dict['coop_roi_labels'] = coop_target_dict['roi_labels']
            # batch_dict['coop_roi_targets_dict'] = coop_target_dict

            # detection target assignment
            targets_dict = self.roi_head.assign_targets(batch_dict)

            # # # concat coop and local targets
            # for key, value in coop_target_dict.items():
            #     targets_dict[key] = torch.cat((targets_dict[key], value), dim=1)
            # # # todo use this in coop proposal? Or BFE?
            
            #----------------------------------
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_targets_dict'] = targets_dict
            num_rois_per_scene = targets_dict['rois'].shape[1]
            if 'roi_valid_num' in batch_dict:
                batch_dict['roi_valid_num'] = [num_rois_per_scene for _ in range(batch_dict['batch_size'])]


        batch_dict = self.pfe(batch_dict)
        if self.model_cfg.PFE.USE_BFE:
            batch_dict = self.bfe(batch_dict)

        batch_dict = self.point_head(batch_dict)

        #always use coop detections as region proposals
        if not self.training and self.model_cfg.get('COOP') is not None and self.model_cfg.COOP.USE_COOP_PROPOSALS_IN_ROI_HEAD:
            batch_dict['rois'] = torch.cat((batch_dict['rois'], batch_dict['coop_boxes']), dim=1)

            #shuffle region proposals
            batch_dict['rois'] = batch_dict['rois'][:, torch.randperm(batch_dict['rois'].shape[1])]


        batch_dict = self.roi_head(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        else:
            loss_point = 0
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
