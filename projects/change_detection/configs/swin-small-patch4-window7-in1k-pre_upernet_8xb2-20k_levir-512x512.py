_base_ = [
    'swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-20k_levir-512x512.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        depths=[2, 2, 18, 2]),
    decode_head=dict(in_channels=[192, 384, 768, 1536], num_classes=2),
    auxiliary_head=dict(in_channels=768, num_classes=2))
