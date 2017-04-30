# test_layer_1.py

from SP_Layer_1 import SP_Layer_1
from scipy import misc


deep_symm = SP_Layer_1()

print('test_sample')
img_list = list()

img_list.append(misc.imread('img_sample/l1_good.png'))
img_list.append(misc.imread('img_sample/l1_not_good.png'))

print(deep_symm.run_network(img_list))
print(deep_symm.run_network(img_list))

# end of script
