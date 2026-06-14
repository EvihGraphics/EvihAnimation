import json
data = json.load(open('Demos/MimicKitReplay/results/white_knight_full_v2/visual_metric_report.json'))
print('Frame 5 IoU:', [x['silhouette_iou'] for x in data['frames_sampled'] if x['frame_id']==5][0])
