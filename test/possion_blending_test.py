import PIL.Image
import numpy as np

from possionblending.possionblending import PossionBlending

figure_dir = '../possionblending/testimages/'
result_dir = '../results/possionblending/'

num_figures = 4
# the row offset and column offset of mask images in target images
mask_offsets = [(29, -37), (149, 126), (22, 9), (0, 33)]
print('# Poisson image blending.')
for i in range(1, num_figures + 1):
    print('# Process image group %d.' % i)
    img_mask = np.asarray(PIL.Image.open(figure_dir + 'test%d_mask.png' % i))
    img_mask.flags.writeable = True
    img_source = np.asarray(PIL.Image.open(figure_dir + 'test%d_src.png' % i))
    img_source.flags.writeable = True
    img_target = np.asarray(PIL.Image.open(figure_dir + 'test%d_target.png' % i))
    img_target.flags.writeable = True

    blend = PossionBlending(img_source, img_mask, img_target, mask_offset=mask_offsets[i - 1])

    # finite difference solver (FDM)
    img_ret_fdm = blend.fdm_solver()
    img_ret = PIL.Image.fromarray(np.uint8(img_ret_fdm))
    img_ret.save(result_dir + 'test%d_fdm.png' % i)
    # finite element solver (FEM)
    img_ret_fem = blend.fem_solver()
    img_ret_fem = PIL.Image.fromarray(np.uint8(img_ret_fem))
    img_ret_fem.save(result_dir + 'test%d_fem.png' % i)
    # direct result
    img_ret_direct = blend.direct_solver()
    img_ret_direct = PIL.Image.fromarray(np.uint8(img_ret_direct))
    img_ret_direct.save(result_dir + 'test%d_direct.png' % i)
