from . import selim_zoo
from . import fpn

models = {
    'selim_dn161_unet': selim_zoo.dn161_unet,
    'selim_dn161_unet_fatter': selim_zoo.dn161_unet_fatter,
    'selim_dn161_sota_unet': selim_zoo.dn161_sota_unet,
    'selim_dn121_unet': selim_zoo.dn121_unet,
    'selim_srx50_unet': selim_zoo.srx50_unet,
    'selim_srx50_unet_dropout': selim_zoo.srx50_unet_dropout,

    'selim_rn50_unet': selim_zoo.rn50_unet,

    'selim_convt_rn50_unet': selim_zoo.convt_rn50_unet,
    'selim_convt_rn34_unet_light': selim_zoo.convt_rn34_unet_light,
    'selim_convt_rn18_unet_light': selim_zoo.convt_rn18_unet_light,

    'selim_sn154_unet': selim_zoo.sn154_unet,
    'selim_pd_dn161_unet': selim_zoo.pd_dn161_unet,
    'selim_pd_rn154_unet': selim_zoo.pd_rn154_unet,
    'selim_rn34_unet': selim_zoo.rn34_unet,
    'selim_rn34_unet_dropout': selim_zoo.rn34_unet_dropout,

    'selim_rn18_unet': selim_zoo.rn18_unet,
    'selim_rx101_unet': selim_zoo.rx101_unet,

    'selim_effnetb0_unet': selim_zoo.effnet_b0_unet,

    'rn50_fpn': fpn.fpn_segmentation,
}
